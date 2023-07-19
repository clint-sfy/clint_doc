# 自定义键盘

参考 插件市场作者 `sola` 的组件进行二次修改优化封装，插件ID：sola-plate-input

> 如果文档内没有找到您需要的帮助，可以尝试[联系作者](https://gitee.com/kevin_chou)

## 平台差异说明
| App | 微信小程序 | 支付宝小程序 | H5浏览器 | 快应用 | 百度小程序 | 字节跳动小程序 | QQ小程序 | 360小程序 | 快手小程序 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| app-vue | ✅ | ❓ | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | 

## 其他说明
 - 兼容 H5、 微信小程序，其他平台未测试。
 - 截图中的车辆号码为随机输入，并非真实车牌号。
 - 基于原作者二次修改：
	1. 页面整体UI设计
	2. 修改了页面适配iphoneX系列刘海屏手机及底部安全区域
	3. 代码精简优化，事件触发优化


## 功能截图
<p align="center">
	<img src="https://zhoukaiwen.com/img/Design/app/FotoJet6.jpg" width="90%" />
</p>

## 功能目录
```vue
.
├─tn_components       	// 组件模板页面入口
	├─keyboard				// ⬅️ 自定义键盘
```

## 使用文档

 - 引入
```js
import plateInput from '@/components/uni-plate-input/uni-plate-input.vue';
```

 - 声明
```js
components: {
	plateInput
},
```

 - 参数
```js
 :plate="plateNo"
```

 - 事件
 ```
 @export //设置车牌号
 @close 	//关闭弹出窗
 ```



## 注意事项
::: warning 注意事项

- 页面icon图标是由[iconfont](https://www.iconfont.cn/collections/detail?cid=29958)下载的，已经上传至我的服务器中。
:::

**如您有其他方面的定制开发，您可以联系作者**


