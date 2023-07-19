# 介绍

## 简介
<p align="center">
    <img width="100" src="../../public/logo.png">
</p>
<p align="center">
	<a href="https://gitee.com/kevin_chou/qdpz/stargazers" target="_blank">
		<img src="https://svg.hamm.cn/gitee.svg?type=star&user=kevin_chou&project=qdpz"/>
	</a>
	<a style="margin:0 10px 0 10px" href="https://gitee.com/kevin_chou/qdpz/members" target="_blank">
		<img src="https://svg.hamm.cn/gitee.svg?type=fork&user=kevin_chou&project=qdpz"/>
	</a>
	<!-- <img src="https://svg.hamm.cn/badge.svg?key=Platform&value=移动端"/> -->
</p>

<h1 align="center" style="text-align:center">《阿源编程》· 开源，易上手~ </h1>


<p align="center">基于uni-app、colorUi、uView、SpringBoot，支持小程序、H5、Android和IOS</p>

<p align="center" style="margin:20px 0 40px 0">
🕙 项目基本保持每日更新，右上随手点个 🌟 Star 关注，这样才有持续下去的动力，谢谢～
</p>

[阿源编程](https://gitee.com/clint_sfy) 是一个基于 [Vue](https://github.com/vuejs/vue-next)、[uniApp](https://uniapp.dcloud.io/)、 [ColorUi](http://demo.color-ui.com/)、[uView](https://www.uviewui.com/) 的web移动端解决方案，它使用了最新的前端技术栈，完美支持微信小程序，包含功能：自定义TabBar与顶部、地图轨迹回放、电子签名、图片编辑器、自定义相机/键盘、拍照图片水印、在线答题、证件识别、周边定位查询、文档预览、各种图表、行政区域、海报生成器、视频播放、主题切换、时间轴、瀑布流、排行榜、课程表、渐变动画、加载动画、请求封装等～ 它可以帮助你快速搭建移动端项目，该项目使用最新的前端技术栈，相信不管是从新技术使用还是其他方面，都能帮助到你。

## 项目体验
<!-- <p align="center">
	<img src="https://cdn.zhoukaiwen.com/qdpz_ewm.png" width="70%" />
</p> -->

## 部分截图
<!-- <p align="center">
	<img src="https://cdn.zhoukaiwen.com/FotoJet0.jpg" width="80%" />
</p>
<p align="center">
	<img src="https://cdn.zhoukaiwen.com/FotoJet2.png" width="50%" />
	<img src="https://zhoukaiwen.com/img/Design/app/FotoJet1.jpg" width="50%" />
</p>
<p align="center">
	<img src="https://cdn.zhoukaiwen.com/FotoJet3.png" width="50%" />
	<img src="https://cdn.zhoukaiwen.com/FotoJet4.png" width="50%" />
</p>
<p align="center">
	<img src="https://zhoukaiwen.com/img/Design/app/FotoJet6.jpg" width="50%" />
	<img src="https://zhoukaiwen.com/img/Design/app/FotoJet7.jpg" width="50%" />
</p>
<p align="center">
	<img src="https://cdn.zhoukaiwen.com/FotoJet5.png" width="50%" />
	<img src="https://zhoukaiwen.com/img/Design/app/FotoJet9.jpg" width="50%" />
</p>
<p align="center">
<img src="https://zhoukaiwen.com/img/Design/app/FotoJet8.jpg" width="50%" />
<img src="https://zhoukaiwen.com/img/Design/app/FotoJet12.jpg" width="50%" />
</p>
<p align="center">
<img src="https://zhoukaiwen.com/img/Design/app/FotoJet13.jpg" width="50%" />
<img src="https://cdn.zhoukaiwen.com/FotoJet13.jpg" width="50%" />
<img src="https://zhoukaiwen.com/img/Design/app/FotoJet11.jpg" width="50%" />
</p> -->

## 目录结构
```                
├─colorui        		// colorui插件依赖
├─common              	// 项目相关公共js方法
│	├─amap-wx.js		// 高德地图依赖js
│	├─classify.data.js	// 模拟数据
│	├─geocode-utils.js	// 腾讯地图方法封装
│	├─projectData.js	// 项目模拟数据
│	├─qqmap-wx-jssdk.js	// 腾讯地图依赖js
│	├─request.js		// 数据请求封装
│	└─uiImg.js			// 模拟数据
│
├─components          	// 项目中使用到的功能封装
│
├─os_project      		// 客户项目入口
│
├─pages      			// 页面入口文件夹
│	├─index				// 主页4个TabBar页面
│	├─me				// 个人中心内页面
│	├─news				// 新闻页
│	├─project			// 项目展示页
│	├─design			// 设计模板 · 瀑布流
│	├─timeline			// 时间轴
│	└─video				// 视频播放页
│
└─video					// 付费模版入口
│	├─customCamera		// 自定义相机/图片编辑器
│	├─posterList		// 海报设计列表
│	└─posterImg			// 海报设计详情页
│
├─static            	// 静态资源
├─tn_components       	// 组件模板页面入口
	├─drag_demo				// 悬浮球
	├─chat					// 聊天室
	├─login					// 登录页合集
	├─photoWall				// 照片墙功能
	├─anloading.vue			// 自定义加载框
	└─bgcolor.vue			// 背景色
	└─bggrad.vue			// 背景渐变
	└─charts.vue			// 图表展示
	└─clock.vue				// 每日签到
	└─company.vue			// 自定义相机
	└─course.vue			// 课班信息
	└─discern.vue			// 证件识别
	└─details.vue			// 通用详情页
	└─district.vue			// 行政区域图
	└─guide.vue				// 引导页
	└─imageEditor.vue		// 图片编辑器
	└─keyboard.vue			// 自定义键盘
	└─mapLocus.vue			// 地图轨迹
	└─medal.vue				// 会员中心
	└─mimicry.vue			// 新拟态
	└─openDocument.vue		// 文档预览
	└─pano.vue				// webview高德地图
	└─poster.vue			// 海报生成器
	└─request.vue			// 模拟数据请求
	└─takePicture.vue		// 摄影师资料
	└─salary.vue			// 排行榜
	└─search.vue			// 便捷查询
	└─sign.vue				// 手写签名
	└─timeline.vue			// 时间轴
	└─timetables.vue		// 课程表
├─uview-ui				// uview-ui插件依赖
├─App.vue				// vue项目入口文件
├─LICENSE				// 许可证
├─main.js				// 公共js
├─manifest.json			// uniapp项目配置文件
├─pages.json			// 页面路由配置页
├─README.md				// 说明文档
└─uni.scss				// uniapp内置的常用样式变量

```
## 运行·前端铺子
> *  注意：运行前删除AppID，重新获取或替换成您的。
- 下载安装：「HBuildX」、「微信开发者工具」
- 扫码登陆微信开发者工具
- 将项目拖进【HBuildX】- 运行 - 微信小程序 - 完成



## 文档

- 中文文档地址为 [阿源编程·官网](https://gitee.com/clint_sfy)，采用 Vitepress 开发。如发现文档有误，欢迎提 pr 帮助我们改进。

### 本地运行文档

如需本地运行文档，请拉取代码到本地。

```bash
# 拉取代码
git clone https://gitee.com/kevin_chou/qdpz.git

# 进入项目目录
cd qdpz-docs

# 安装依赖
yarn

# 运行项目
yarn dev
```

## 基础知识

本项目需要一定前端基础知识，请确保掌握 Vue/uniApp 的基础知识，以便能处理一些常见的问题。
建议在开发前先学一下以下内容，提前了解和学习这些知识，会对项目理解非常有帮助:

- [Vue 官网](https://cn.vuejs.org/)
- [uniApp 官网](https://uniapp.dcloud.io/)
- [ColorUi](http://demo.color-ui.com/)
- [uView](https://www.uviewui.com/)
- [Es6](https://es6.ruanyifeng.com/)

## 浏览器支持

**本地开发**推荐使用`HBuilder`编辑器 ➕ `微信开发者工具`，HBuilder **不支持**`2.0`以下版本。

| App | 微信小程序 | 支付宝小程序 | QQ小程序 | H5浏览器 | 微信公众号 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 需nvue | ✅ | ✅ | ✅ | ✅ | ✅ |

## 作者信息
- 姓名：申丰源 (clint-sfy)
- 微信：-clint
- 邮箱：2786435349@qq.com

## 加入我们

- [阿源编程](https://gitee.com/clint_sfy) 还在持续更新中，本项目欢迎您的参与，共同维护。
- 如果你想加入我们，可以多提供一些好的建议或者提交 pr，我们会根据你的活跃度邀请你加入。
- 加入技术交流群，请备注信息:「前端铺子」，群聊已加入 图鸟-可我会像、TopicQ作者等等前后端全栈大佬

<!-- <img src="https://zhoukaiwen.com/img/qdpz/wxq.png" width="45%" /> -->
