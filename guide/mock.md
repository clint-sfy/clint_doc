# 数据&联调

## 如何使用

::: tip 说明：

- [前端铺子/common/request.js](https://gitee.com/kevin_chou/qdpz/blob/develop/common/request.js) 已制作封装数据请求，并在[前端铺子/tn_components/request.vue](https://gitee.com/kevin_chou/qdpz/blob/develop/tn_components/request.vue) 提供请求数据的案例页面，您可以在任何项目中随意使用。
- 请求封装已制作：含Token请求/不含Token请求的两种方式
- 请求可复制到其他uniapp项目中使用
- 包含“全局拦截”
- 案例请求并非真实接口，而是请求了服务器上的Json文件，您也可以复制测试使用，压测无用哦

:::

如果您是微信小程序，请配置您的接口域名并且使用https方式请求，否则上线后无法使用

### 配置

修改您接口前缀
```bash
# 替换成您的接口前缀
const baseUrl = 'https://www.zhoukaiwen.com/';
```


### 不带Token请求

```js
const httpRequest = (opts, data) => {
	uni.onNetworkStatusChange(function(res) {
		if (!res.isConnected) {
			uni.showToast({
				title: '网络连接不可用！',
				icon: 'none'
			});
		}
		return false
	});
	let httpDefaultOpts = {
		url: baseUrl + opts.url,
		data: data,
		method: opts.method,
		header: opts.method == 'get' ? {
			'X-Requested-With': 'XMLHttpRequest',
			"Accept": "application/json",
			"Content-Type": "application/json; charset=UTF-8"
		} : {
			'X-Requested-With': 'XMLHttpRequest',
			'Content-Type': 'application/json; charset=UTF-8'
		},
		dataType: 'json',
	}
	let promise = new Promise(function(resolve, reject) {
		uni.request(httpDefaultOpts).then(
			(res) => {
				resolve(res[1])
			}
		).catch(
			(response) => {
				reject(response)
			}
		)
	})
	return promise
};
```

### 含Token请求

```js
const httpTokenRequest = (opts, data) => {
	uni.onNetworkStatusChange(function(res) {
		if (!res.isConnected) {
			uni.showToast({
				title: '网络连接不可用！',
				icon: 'none'
			});
		}
		return false
	});
	let token = uni.getStorageSync('token');
	// hadToken()
	if (token == '' || token == undefined || token == null) {
		uni.showToast({
			title: '账号已过期，请重新登录',
			icon: 'none',
			complete: function() {
				uni.reLaunch({
					url: '/pages/login/index'
				});
			}
		});
	} else {
		let httpDefaultOpts = {
			url: baseUrl + opts.url,
			data: data,
			method: opts.method,
			header: opts.method == 'get' ? {
				'X-Access-Token': token,
				'X-Requested-With': 'XMLHttpRequest',
				"Accept": "application/json",
				"Content-Type": "application/json; charset=UTF-8"
			} : {
				'X-Access-Token': token,
				'X-Requested-With': 'XMLHttpRequest',
				'Content-Type': 'application/json; charset=UTF-8'
			},
			dataType: 'json',
		}
		let promise = new Promise(function(resolve, reject) {
			uni.request(httpDefaultOpts).then(
				(res) => {
					if (res[1].data.code == 200) {
						resolve(res[1])
					} else {
						if (res[1].data.code == 5000) {
							 uni.showModal({
							 	title: 'Token已过期',
							 	content: res[1].data.message,
							 	success: function (res) {
							 		if (res.confirm) {
							 			uni.reLaunch({
							 				url: '/pages/login/login'
							 			});
							 			uni.clearStorageSync();
							 		} 
							 	}
							 });
							uni.reLaunch({
								url: '/pages/login/index'
							});
							uni.clearStorageSync();
						} else {
							resolve(res[1])
							// uni.showToast({
							// 	title: '' + res[1].data.message,
							// 	icon: 'none'
							// })
						}
					}
				}
			).catch(
				(response) => {
					reject(response)
				}
			)
		})
		return promise
	}
	// let token = uni.getStorageSync('token')
	//此token是登录成功后后台返回保存在storage中的
};
```
::: tip 注意

 **`此token是登录成功后后台返回保存在storage中的`**
 
 **`Token请求拦截`**，您可随意配置跳转到您项目的任何页面或弹窗提醒。

:::

### 设置请求拦截

```js
const hadToken = () => {
	let token = uni.getStorageSync('token');

	if (token == '' || token == undefined || token == null) {
		uni.showToast({
			title: '账号已过期，请重新登录',
			icon: 'none',
			complete: function() {
				uni.reLaunch({
					url: '/pages/login/index'
				});
			}
		});
		return false;
	}
	return true
}
```

## 请求案例
```js
# 引入js
import request from '@/common/request.js';

let opts = {
	url: 'json/project.json',
	method: 'get'
};
request.httpRequest(opts).then(res => {
	console.log(res);
	uni.hideLoading();
	if(res.statusCode == 200){
		this.dataList = res.data;
	}else{
		console.log('请求失败')
	}
});

```

::: tip 关于Mock

当然项目中可以使用 `Mock.js` 模拟数据，此项目因考虑便捷性没有引入，如这样会增加打包体积，如果您有需要请联系作者，给您提供完整包含Mock的项目Demo。

:::

