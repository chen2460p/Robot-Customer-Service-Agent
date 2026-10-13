"""
================================================================================
文件：agent/tools/weather_client.py
职责：真实天气查询客户端——调用 Open-Meteo 公开 API（**免密钥**，开箱即用）。

为什么选 Open-Meteo 而不是和风/OpenWeather？
    - 后两者都需要注册申请 API Key，别人 clone 这个项目后跑不起来、也无法当场演示；
    - Open-Meteo 免密钥、无需注册，且同时提供【地理编码】接口，
      能把"深圳"这种城市名转成经纬度，正好让这个工具具备真实工程价值。

工程要点（这四点才是这个模块值得写进简历的地方）：
    1. 超时控制：每次请求都带 connect/read 超时，避免网络抖动时整个 Agent 卡死。
    2. 失败重试：指数退避（0.5s → 1s → 2s），应对偶发网络错误而非立刻放弃。
    3. 两级 TTL 缓存：天气结果短缓存（默认 600s），城市经纬度长缓存（默认 1 天）——
       因为经纬度几乎不变，而天气会变。实测冷调用地理编码约 1.3s，命中缓存后为 0ms。
    4. 优雅降级：任何异常都被收敛为返回 None，由上层决定降级策略；
       空气质量属可选增强，该接口失败时仍正常返回主天气信息，不影响主流程。

与谁配合：
    - 被 agent/tools/agent_tools.py 的 get_weather 工具调用。
    - 读取 config/agent.yml 的 weather 段配置（超时/重试/缓存时长/降级文案）。
    - 复用 utils/logger_handler.py 的统一日志器。
================================================================================
"""

import threading
import time

import requests

from utils.config_handler import agent_conf
from utils.logger_handler import logger

# WMO 标准天气代码 → 中文描述
# Open-Meteo 的 weather_code 采用 WMO 4677 标准，这里只映射常用代码，
# 未收录的代码会被格式化成"未知天气(代码 N)"，不会因为漏配而报错。
WMO_CODE_MAP = {
    0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    56: "轻度冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "轻度冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴小冰雹", 99: "雷阵雨伴大冰雹",
}


class WeatherClient:
    """天气查询客户端：带超时 / 重试 / 两级缓存 / 降级，对外只暴露 get_weather。"""

    def __init__(self, conf: dict | None = None):
        """
        :param conf: 可选的配置字典，用于测试时注入（默认读 config/agent.yml 的 weather 段）
        """
        # 允许外部注入配置，方便单测替换成"坏地址"来验证降级逻辑
        conf = conf if conf is not None else (agent_conf.get("weather") or {})

        self.geocode_url = conf.get("geocode_url", "https://geocoding-api.open-meteo.com/v1/search")
        self.forecast_url = conf.get("forecast_url", "https://api.open-meteo.com/v1/forecast")
        self.air_quality_url = conf.get("air_quality_url",
                                        "https://air-quality-api.open-meteo.com/v1/air-quality")
        self.timeout = float(conf.get("timeout", 8))          # 单次请求超时（秒）
        self.retries = int(conf.get("retries", 3))            # 最大尝试次数
        self.cache_ttl = float(conf.get("cache_ttl", 600))    # 天气结果缓存时长（秒）
        self.geo_cache_ttl = float(conf.get("geo_cache_ttl", 86400))   # 经纬度缓存时长（秒）
        self.fallback_text = conf.get("fallback_text", "天气服务暂时不可用，请稍后再试。")

        # 复用 Session：保持 TCP 连接与 TLS 会话，省掉重复握手的开销
        self._session = requests.Session()

        # 两级缓存：key 均为城市名 -> (过期时间戳, 值)
        self._cache: dict[str, tuple[float, str]] = {}                 # 城市 -> 天气文本
        self._geo_cache: dict[str, tuple[float, float, float, str]] = {}  # 城市 -> (过期, 纬度, 经度, 标准名)

        # Streamlit 的会话可能并发触发查询，加锁保证缓存读写安全
        self._lock = threading.Lock()

        # 运行统计：用于观测缓存收益（缓存命中率）与接口健康状况
        self.stats = {"request": 0, "cache_hit": 0, "retry": 0, "failure": 0}

    # ------------------------------------------------------------------
    # 内部：带指数退避重试的 GET
    # ------------------------------------------------------------------
    def _get_json(self, url: str, params: dict) -> dict:
        """
        发起 GET 并解析 JSON；失败按指数退避重试，尝试次数用尽后抛出最后一次异常。

        为什么重试要用【指数退避】而不是固定间隔？
            网络抖动通常是瞬时的，退避能让请求错开，避免在服务端刚出问题时
            以固定频率持续打过去，反而加剧拥塞。
        """
        delay = 0.5
        last_exc: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                # raise_for_status：把 4xx/5xx 也转成异常，统一走下面的重试/降级逻辑
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                self.stats["retry"] += 1
                logger.warning(f"[weather] 第 {attempt}/{self.retries} 次请求失败 {url} -> "
                               f"{type(e).__name__}: {e}")
                # 最后一次失败就不要再睡了，直接退出循环抛异常
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2

        raise last_exc

    # ------------------------------------------------------------------
    # 内部：地理编码（城市名 -> 经纬度），带长缓存
    # ------------------------------------------------------------------
    def _geocode(self, city: str) -> tuple[float, float, str] | None:
        """把城市名解析为 (纬度, 经度, 标准城市名)；查不到返回 None。"""
        now = time.time()
        with self._lock:
            hit = self._geo_cache.get(city)
            if hit and hit[0] > now:
                return hit[1], hit[2], hit[3]

        data = self._get_json(self.geocode_url,
                              {"name": city, "count": 1, "language": "zh", "format": "json"})
        results = data.get("results") or []
        if not results:
            # 城市名无法识别（如错别字、不存在的地名）——这不是异常，是业务上的"没找到"
            logger.warning(f"[weather] 地理编码未匹配到城市：{city}")
            return None

        top = results[0]
        lat, lon = float(top["latitude"]), float(top["longitude"])
        # 用接口返回的标准名（如用户传"深圳"→标准名"深圳"，传拼音也会返回规范中文名）
        name = top.get("name") or city

        with self._lock:
            self._geo_cache[city] = (now + self.geo_cache_ttl, lat, lon, name)
        return lat, lon, name

    # ------------------------------------------------------------------
    # 内部：空气质量等级（欧洲 AQI 分档）
    # ------------------------------------------------------------------
    @staticmethod
    def _aqi_level(aqi: float) -> str:
        """把欧洲 AQI 数值映射为中文等级。"""
        for limit, label in ((20, "优"), (40, "良"), (60, "中等"), (80, "较差"), (100, "很差")):
            if aqi <= limit:
                return label
        return "极差"

    # ------------------------------------------------------------------
    # 内部：拉取并拼装天气文本
    # ------------------------------------------------------------------
    def _fetch_and_format(self, lat: float, lon: float, name: str) -> str:
        """按经纬度取实时天气（含未来 6 小时降雨概率）与空气质量，拼成一句中文描述。"""
        self.stats["request"] += 1

        forecast = self._get_json(self.forecast_url, {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "hourly": "precipitation_probability",
            "forecast_days": 1,
            "timezone": "auto",     # 按当地时区返回，避免用 UTC 取错小时
        })

        cur = forecast.get("current", {})
        code = cur.get("weather_code")
        # 未收录的代码降级成"未知天气(代码N)"，保证不会因映射缺失而中断
        desc = WMO_CODE_MAP.get(code, f"未知天气(代码{code})")

        # hourly 是逐小时数组；取前 6 项即"未来 6 小时"，
        # 用【最大值】而不是平均值，因为对"要不要现在开机器人"来说，最坏情况更有决策价值
        probs = (forecast.get("hourly", {}).get("precipitation_probability") or [])[:6]
        prob_text = f"未来6小时最大降雨概率{max(probs)}%" if probs else "降雨概率暂无数据"

        # 空气质量属于【可选增强】：单独 try，失败不影响主天气信息返回
        aqi_text = ""
        try:
            air = self._get_json(self.air_quality_url, {
                "latitude": lat, "longitude": lon,
                "current": "european_aqi,pm2_5", "timezone": "auto",
            })
            a = air.get("current", {})
            aqi = a.get("european_aqi")
            if aqi is not None:
                aqi_text = (f"，空气质量AQI {aqi}（{self._aqi_level(aqi)}），"
                            f"PM2.5 {a.get('pm2_5')}微克/立方米")
        except Exception as e:
            logger.warning(f"[weather] 空气质量获取失败，降级为仅返回天气信息："
                           f"{type(e).__name__}: {e}")

        return (f"城市{name}天气为{desc}，气温{cur.get('temperature_2m')}摄氏度，"
                f"空气湿度{cur.get('relative_humidity_2m')}%，"
                f"风速{cur.get('wind_speed_10m')}公里/小时{aqi_text}，{prob_text}")

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def get_weather(self, city: str) -> str | None:
        """
        查询指定城市的实时天气。

        :param city: 城市名（中文 / 拼音均可，由地理编码接口识别）
        :return: 中文天气描述；**任何失败一律返回 None**，由调用方决定降级策略
                 （不在这里抛异常，是为了不让天气这个非核心工具拖垮整个 Agent 流程）
        """
        if not city or not city.strip():
            return None
        city = city.strip()

        now = time.time()
        with self._lock:
            hit = self._cache.get(city)
            if hit and hit[0] > now:
                self.stats["cache_hit"] += 1
                logger.info(f"[weather] 命中缓存，跳过网络请求：{city}")
                return hit[1]

        try:
            geo = self._geocode(city)
            if geo is None:
                return None
            text = self._fetch_and_format(*geo)
        except Exception as e:
            self.stats["failure"] += 1
            logger.error(f"[weather] 查询失败并降级：{city} -> {type(e).__name__}: {e}")
            return None

        with self._lock:
            self._cache[city] = (now + self.cache_ttl, text)
        return text

    def cache_hit_rate(self) -> float:
        """缓存命中率（%）——用于量化"缓存到底省掉了多少次网络请求"。"""
        total = self.stats["cache_hit"] + self.stats["request"]
        return self.stats["cache_hit"] / total * 100 if total else 0.0


if __name__ == '__main__':
    # 自测：验证真实查询、缓存加速、以及降级（把地址指向不存在的域名）
    client = WeatherClient()

    for city in ["深圳", "合肥"]:
        t0 = time.perf_counter()
        print(f"{city} -> {client.get_weather(city)}")
        print(f"   首次(冷) 耗时 {(time.perf_counter() - t0) * 1000:.0f} ms")

    t0 = time.perf_counter()
    client.get_weather("深圳")
    print(f"深圳 -> 二次(缓存) 耗时 {(time.perf_counter() - t0) * 1000:.0f} ms")
    print(f"缓存命中率: {client.cache_hit_rate():.0f}%  统计: {client.stats}")

    # 降级验证：超时设为 0.001s 必然失败，应返回 None 而不是抛异常
    bad = WeatherClient({**dict(agent_conf.get("weather") or {}), "timeout": 0.001, "retries": 2})
    print(f"降级验证 -> {bad.get_weather('深圳')}   统计: {bad.stats}")
