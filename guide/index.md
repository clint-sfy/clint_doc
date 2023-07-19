# 开始

本文会帮助你从头启动项目

## 前言

::: tip 关于组件

项目虽然二次封装了一些组件，但是可能不能满足大部分的要求。所以，如果组件不满足你的要求，完全可以不用甚至删除代码自己写，不必坚持使用项目自带的组件。

:::

## 环境准备

本地环境建议安装 [npm](https://www.npmjs.com/)、[Node.js](http://nodejs.org/) 和 [Git](https://git-scm.com/)

::: warning 注意

- 建议安装能使您开发更方便快捷，并非必须安装（因HBuilder内包含环境编译）

:::

## 工具配置

如果您使用的 HBuilder 是[HBuilderX](https://www.dcloud.io/hbuilderx.html)(推荐)的话，可以安装以下工具来提高开发效率及代码格式化

- [App真机运行]
- [eslint-plugin-vue]
- [Git插件]
- [less插件]
- [scss/sass编译]
- [stylus编译]
- [typescript语言服务]
- [uni-app编译]
- [uni_helpers]
- [uni_modules插件]
- [uniCloud本地调试运行]
- [内置浏览器]
- [内置终端]

## 代码获取

### 从 Gitee 获取代码

因 github clone 代码较慢，您可以尝试用 [Gitee](https://gitee.com/kevin_chou/qdpz.git) 同步代码到自己的仓库，再 clone 下来即可。

也可以通过下方地址进行 clone

```bash
git clone https://gitee.com/kevin_chou/qdpz.git
```

## 安装

### 安装 Node.js

如果您电脑未安装[Node.js](https://nodejs.org/en/)，请安装它。

**验证**

```bash
# 出现相应npm版本即可
npm -v
# 出现相应node版本即可
node -v

```

如果你需要同时存在多个 node 版本，可以使用 [Nvm](https://github.com/nvm-sh/nvm) 或者其他工具进行 Node.js 进行版本管理。

### 安装 微信开发者工具
如果您电脑未安装[微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/stable.html)，请安装它。

接下来你可以修改代码进行业务开发了。我们内建了Mock/Json模拟数据、热更新、自定义头部/底部、全局路由、分包等各种实用的功能辅助开发，请阅读其他章节了解更多。

## 目录说明

```bash

.
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
├─pages      			// 页面入口文件夹
│	├─index				// 主页4个TabBar页面
│	├─me				// 个人中心内页面
│	├─news				// 新闻页
│	├─project			// 项目展示页
│	├─design			// 设计模板 · 瀑布流
│	├─timeline			// 时间轴
│	└─video				// 视频播放页
│
├─static            	// 静态资源
├─tn_components       	// 组件模板页面入口
├─uview-ui				// uview-ui插件依赖
├─App.vue				// vue项目入口文件
├─LICENSE				// 许可证
├─main.js				// 公共js
├─manifest.json			// uniapp项目配置文件
├─pages.json			// 页面路由配置页
├─README.md				// 说明文档
└─uni.scss				// uniapp内置的常用样式变量

```
