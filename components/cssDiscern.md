# 证件识别

页面导航栏、顶部栏、顶部提醒等部分参考 `colorUi/uView` 实现，样式组件不包含逻辑代码

## 页面截图
截图中的人员信息为随机输入，并非真实。
<p align="center">
	<img src="https://zhoukaiwen.com/img/Design/app/zjx_4.png" width="40%" />
</p>

## 页面目录
```vue
.
├─tn_components       	// 组件模板页面入口
	├─discern				// ⬅️ 证件识别
```
## 其他说明
 - 兼容 H5、 微信小程序、App，其他平台未测试。
 - 截图中的人员信息为随机输入，并非真实。
 - 基于原作者二次修改可接入百度/阿里人脸识别，页面仅提供样式

**页面上传图片已制作，您可以上传图片至后台比对库**

```js
uploadImg() {
	uni.chooseImage({
		count: 1,
		success: (chooseImageRes) => {
			const tempFilePaths = chooseImageRes.tempFilePaths;
			console.log(chooseImageRes);
			uni.showToast({
				icon: 'none',
				title: '上传成功，暂无接口预览',
				duration: 2000
			});
			return false;
			uni.uploadFile({
				url: https://www.zhoukaiwen.com, //仅为示例，非真实的接口地址
				filePath: tempFilePaths[0],
				name: 'file',
				header: {
					"Content-Type": "multipart/form-data",
					'X-Access-Token': uni.getStorageSync('token'),
				},
				success: (uploadFileRes) => {
					this.form.userBaseInfo.headPhoto = JSON.parse(uploadFileRes.data).message
				}
			});
		}
	});
}
```

**如您有其他方面的定制开发，您可以联系作者**


