# 登录合集
账号密码登录共4个页面，微信授权登录1个页面（图鸟设计）

> 页面中数据为模拟数据，并非真实数据

## 页面截图
<p align="center">
	<img src="https://zhoukaiwen.com/img/Design/app/FotoJet11.jpg" width="60%" />
</p>
<p align="center" style="color:#111;font-size:22px">账号密码登录</p>

<p align="center" style="margin-top:60px">
	<img src="https://cdn.zhoukaiwen.com/qdpz_login19.jpg" width="20%" />
</p>
<p align="center" style="color:#111;font-size:22px;">微信授权登录</p>

## 页面目录
```vue
.
└─tn_components       	// 组件模板页面入口
	└─login				// 登录合集
		└─wxLogin.vue	// 微信授权登录
	
```

::: tip 温馨提醒

	静态页面展示，对接自己的后端数据即可使用，此页面主要介绍【微信登录逻辑】

:::

## 授权登录 · 代码
```vue
methods: {
	<!-- 授权小程序 -->
	getUserProfile(){
		var that = this;
		return new Promise((resolve, reject) => {
			uni.getUserProfile({
				desc: '获取您的微信信息以授权小程序',
				success: userRes => {
					console.log('getUserProfile-res', userRes);
					resolve(userRes);
					that.UserProfileRes = userRes
				},
				fail: userErr => {
					uni.showToast({
						title: '授权失败',
						icon: 'error'
					});
					console.log('getUserProfile-err', userErr);
					reject(userErr);
				}
			});
		});
	},
	
	<!-- 注意： -->
	<!-- 注意： -->
	<!-- getLoginCode() & getUserProfile()  二者不可同步调用！！！ -->
	<!-- 注意： -->
	<!-- 注意： -->
	
	<!-- 获取code -->
	getLoginCode(){
		var that = this;
		return new Promise((resolve, reject) => {
			uni.login({
				provider: 'weixin',
				success: loginRes => {
					console.log('loginRes', loginRes);
					that.loginRes = loginRes
					resolve(loginRes);
				}
			});
		});
	},
	<!-- 微信用户授权 -->
	openWxLogin(){
		var that = this;
		uni.showLoading({
			title: '加载中'
		})
		let userProFile = this.getUserProfile();
		let loginCode = this.getLoginCode();
		loginCode.then(code => {
				return code;
			}).then(logCode => {
				return new Promise((resolve, reject) => {
					userProFile.then(res => {
							<!-- 你的请求 -->
						})
						.catch(err => {
							reject(err);
						});
				});
			})
			.then(res => {
				console.log('promise-res', res);
			})
			.catch(err => {
				console.log('userProfile-err', err);
			});
	},
}

```