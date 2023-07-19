# 聊天室
参考 插件市场作者 `回梦無痕` 的组件进行二次修改优化封装

> 如果文档内没有找到您需要的帮助，可以尝试[联系作者](https://ext.dcloud.net.cn/plugin?id=324)

## 平台差异说明
| App | 微信小程序 | 支付宝小程序 | H5浏览器 | 快应用 | 百度小程序 | 字节跳动小程序 | QQ小程序 | 360小程序 | 快手小程序 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 

## 其他说明
 - 模板没有进行网络数据交互，所有消息的收发是由代码本地模拟出来的效果等。
 - 语音录制目前是设置mp3格式
 - 基于原作者二次修改：
	1. 修改了自定义emoji图标
	2. 修改了页面适配iphoneX系列刘海屏手机
	3. 适配优化了底部及红包弹窗的样式


## 功能截图
<p align="center">
	<img src="https://zhoukaiwen.com/img/Design/app/FotoJet10.jpg" width="90%" />
</p>

## 功能目录
```vue
.
├─tn_components       	// 组件模板页面入口
	├─chat				// ⬅️ 聊天室
```

## 注意事项
::: warning 注意事项

- emoji图标是由[iconfont](https://www.iconfont.cn/collections/detail?cid=29958)下载的，已经上传至我的服务器中。
- 服务器图片地址：[https://zhoukaiwen.com/img/icon/emojj1/1.png](https://zhoukaiwen.com/img/icon/emojj1/1.png)
- iOS端的微信小程序中，有些动画貌似有些问题，暂未修复(代码写了一大堆了，最后测微信小程序端发现动画有问题 (0..0)，先搁置吧，以后更新再修复)
:::

**如您有其他方面的定制开发，您可以联系作者**


