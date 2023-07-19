# 地图轨迹回放
对 uniapp `Map地图应用` 组件进行封装

> 如果文档内没有找到您需要的帮助，可以尝试[联系作者](https://gitee.com/kevin_chou)

## 平台差异说明
| App | 微信小程序 | 支付宝小程序 | H5浏览器 | 快应用 | 百度小程序 | 字节跳动小程序 | QQ小程序 | 360小程序 | 快手小程序 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 需nvue | ✅ | ✅ | ✅ | ✅ | ✅ | 1.63+ | 1.9.0+ | ❌ | ❌ | 

**注意：** 如您打包App端，请修改代码为nvue，可参考下载群友开发的 [nvue案例](https://ext.dcloud.net.cn/plugin?id=5680)

## 基础使用
下面是轨迹回放的代码示例
```vue
<template>
	<view>
		<map id='map' :latitude="latitude" :longitude="longitude" :markers="covers" :style="{ width: '100%', height: mapHeight + 'px' }"
		 :scale="13" :polyline="polyline">
		</map>
	</view>
</template>
<script>
	export default {
		data() {
			return {
				map:null,
				
				windowHeight: 0,
				mapHeight: 0,
				timer: null,
				
				isDisabled:false,
				isStart: false,
				playIndex:1,
				
				id: 0, // 使用 marker点击事件 需要填写id
				title: 'map',
				latitude: 34.263734,
				longitude: 108.934843,
				// 标记点
				covers: [{
					id: 1,
					width: 42,
					height: 47,
					rotate: 270,
					latitude: 34.259428,
					longitude: 108.947040,
					iconPath: 'http://cdn.zhoukaiwen.com/car.png',
					callout: {
						content: "陕A·88888", // 车牌信息
						display: "ALWAYS",
						fontWeight: "bold",
						color: "#5A7BEE", //文本颜色
						fontSize: "12px",
						bgColor: "#ffffff", //背景色
						padding: 5, //文本边缘留白
						textAlign: "center",
					},
					anchor: {
						x: 0.5,
						y: 0.5,
					},
				}],
				// 线
				polyline: [],
				// 坐标数据
				coordinate: [{
						latitude: 34.259428,
						longitude: 108.947040,
						problem: false,
					},
					{
						latitude: 34.252918,
						longitude: 108.946963,
						problem: false,
					},
					{
						latitude: 34.252408,
						longitude: 108.946240,
						problem: false,
					},
					{
						latitude: 34.249286,
						longitude: 108.946184,
						problem: false,
					},
					{
						latitude: 34.248670,
						longitude: 108.946640,
						problem: false,
					}
				],
				posi: { // 汽车定位点的数据，后面只用改latitude、longitude即可
					id: 1,
					width: 32,
					height: 32,
					latitude: 0,
					longitude: 0,
					iconPath: "http://cdn.zhoukaiwen.com/car.png",
					callout: {
						content: "陕A·85Q1Q", // 车牌信息
						display: "BYCLICK",
						fontWeight: "bold",
						color: "#5A7BEE", //文本颜色
						fontSize: "12px",
						bgColor: "#ffffff", //背景色
						padding: 5, //文本边缘留白
						textAlign: "center",
					},
					anchor: {
						x: 0.5,
						y: 0.5,
					},
				}
			}
		},
		onReady() {
			// 创建map对象
			this.map = uni.createMapContext('map');
			// 获取屏幕高度
			uni.getSystemInfo({
				success: res => {
					this.windowHeight = res.windowHeight;
				}
			});
		},
		mounted() {
			this.setNavTop('.navBox')

			this.polyline = [{
				points: this.coordinate,
				color: '#025ADD',
				width: 4,
				dottedLine: false,
			}];
		},
		methods: {
			setNavTop(style) {
				let view = uni.createSelectorQuery().select(style);
				view
					.boundingClientRect(data => {
						console.log("tabInList基本信息 = " + data.height);
						this.mapHeight = this.windowHeight - data.height;
						console.log(this.mapHeight)
					})
					.exec();
			},
			start() {
				this.isStart = true;
				this.isDisabled = true;
				let data = this.coordinate;
				let len = data.length;
				let datai = data[this.playIndex];
				let _this = this;
				
				_this.map.translateMarker({
					markerId: 1,
					autoRotate:true,
					destination: {
						longitude: datai.longitude, // 车辆即将移动到的下一个点的经度
						latitude: datai.latitude, // 车辆即将移动到的下一个点的纬度
					},
					duration: 700,
					complete: function(){
						_this.playIndex++;
						if(_this.playIndex < len){
							_this.start(_this.playIndex, data);
						}else{
							console.log('okokok');
							uni.showToast({
								title: '播放完成',
								duration: 1400,
								icon: 'none'
							});
							_this.playIndex = 0;
							_this.isStart = false;
							_this.isDisabled = false;
						}
					},
					animationEnd: function() {
						// 轨迹回放完成 处理H5端
						_this.playIndex++;
						if(_this.playIndex < len){
							_this.start(_this.playIndex, data);
						}else{
							console.log('okokok');
							uni.showToast({
								title: '播放完成',
								duration: 1400,
								icon: 'none'
							});
							_this.playIndex = 0;
							_this.isStart = false;
							_this.isDisabled = false;
						}
					},
					fail(e) {
						// 轨迹回放失败
					},
				});
			},
		}
	}
</script>
```
::: warning 注意事项

- `<map>` 组件的宽/高推荐写直接量，比如：750rpx，不要设置百分比值。
- uni-app 只支持 gcj02 坐标
- App平台 layer-style 属性需要在地图服务商后台创建，值设置为高德后台申请的字符串，[详情](https://developer.amap.com/api/android-sdk/guide/create-map/custom)
- 小程序和app-vue中，`<map>` 组件是由引擎创建的原生组件，它的层级是最高的，不能通过 z-index 控制层级。在`<map>`上绘制内容，可使用组件自带的marker、controls等属性，也可以使用`<cover-view>`组件。App端还可以使用plus.nativeObj.view 或 subNVue 绘制原生内容，参考。另外App端nvue文件不存在层级问题。从微信基础库2.8.3开始，支持map组件的同层渲染，不再有层级问题。
- App端nvue文件的map和小程序拉齐度更高。vue里的map则与plus.map功能一致，和小程序的地图略有差异。**App端使用map推荐使用nvue**。
:::


**如遇到其他问题，建议您仔细阅读uniapp的 [Map说明](https://uniapp.dcloud.io/component/map)**

**如您有其他地图方面的定制开发，您可以联系作者**


