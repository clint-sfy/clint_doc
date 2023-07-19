# 预览&构建&部署

::: tip 前言
uni-app支持通过 **可视化界面** 、**vue-cli命令行** 两种方式快速创建项目

由于前端铺子是开源项目，所以已经去除掉了项目中的 [AppID](https://gitee.com/kevin_chou/qdpz/blob/develop/manifest.json)， 故无法打包小程序，如您需要打包请申请小程序AppID。
:::

## 通过可视化方式

开始之前，开发者需先下载安装如下工具：

- HBuilderX：[官方IDE下载地址](https://www.dcloud.io/hbuilderx.html)

HBuilderX是通用的前端开发工具，但为uni-app做了特别强化

下载App开发版，可开箱即用；如下载标准版，在运行或发行uni-app时，会提示安装uni-app插件，插件下载完成后方可使用

### 运行uni-app

- **浏览器运行** ：进入hello-uniapp项目，点击工具栏的运行 -> 运行到浏览器 -> 选择浏览器，即可在浏览器里面体验uni-app 的 H5 版

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-uni-app-doc/1ad34710-4f1a-11eb-8ff1-d5dcf8779628.png" width="40%" />


- **真机运行** ：连接手机，开启USB调试，进入hello-uniapp项目，点击工具栏的运行 -> 真机运行 -> 选择运行的设备，即可在该设备里面体验uni-app

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-uni-app-doc/3a1faaf0-4f1a-11eb-b680-7980c8a877b8.png" width="40%" />

如 `手机无法识别`，请点击菜单运行-运行到手机或模拟器-真机运行常见故障排查指南。 注意目前开发App也需要安装微信开发者工具

- **在微信开发者工具里运行** ：进入hello-uniapp项目，点击工具栏的运行 -> 运行到小程序模拟器 -> 微信开发者工具，即可在微信开发者工具里面体验uni-app

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-uni-app-doc/d89fd6f0-4f1a-11eb-97b7-0dc4655d6e68.png" width="40%" />

::: warning 注意事项

如果是第一次使用，需要先配置小程序ide的相关路径，才能运行成功。如下图，需在输入框输入微信开发者工具的安装路径。 若HBuilderX不能正常启动微信开发者工具，需要开发者手动启动，然后将uni-app生成小程序工程的路径拷贝到微信开发者工具里面，在HBuilderX里面开发，在微信开发者工具里面就可看到实时的效果。
:::

uni-app默认把项目编译到根目录的unpackage目录

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-uni-app-doc/a142b6a0-4f1a-11eb-8a36-ebb87efcf8c0.png" width="100%" />


- **支付宝小程序开发者工具里运行** ：进入hello-uniapp项目，点击工具栏的运行 -> 运行到小程序模拟器 -> 支付宝小程序开发者工具，即可在支付宝小程序开发者工具里面体验uni-app

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-uni-app-doc/fee90480-4f1a-11eb-bd01-97bc1429a9ff.png" width="40%" />

- 在 **`百度开发者工具`、`字节跳动开发者工具`、`快应用联盟`、`华为开发者工具`、`QQ小程序`**、开发工具里运行：内容同上，不再重复。

::: warning 注意事项
- 如果是第一次使用，需要配置开发工具的相关路径。点击工具栏的运行 -> 运行到小程序模拟器 -> 运行设置，配置相应小程序开发者工具的路径
- 支付宝/百度/字节跳动/360小程序工具，不支持直接指定项目启动并运行。因此开发工具启动后，请将 HBuilderX 控制台中提示的项目路径，在相应小程序开发者工具中打开
- 如果自动启动小程序开发工具失败，请手动启动小程序开发工具并将 HBuilderX 控制台提示的项目路径，打开项目
:::

### 打包uni-app

- **打包为原生App：** 在HBuilderX工具栏，点击发行，选择原生app-云端打包，如下图：

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-uni-app-doc/b8332fd0-4f37-11eb-8ff1-d5dcf8779628.png" width="24%" />

出现如下界面，点击打包即可

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-dc-site/001a20b0-d85a-11ea-81ea-f115fe74321c.png" width="40%" />


- **打包为H5：** 在 manifest.json 的可视化界面，进行如下配置（发行在网站根目录可不配置应用基本路径），此时发行网站路径是 www.xxx.com/h5：

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-uni-app-doc/bf90de30-4f37-11eb-8ff1-d5dcf8779628.png" width="40%" />

在HBuilderX工具栏，点击发行，选择网站-H5手机版，如下图，点击即可生成 H5 的相关资源文件，保存于 unpackage 目录

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-uni-app-doc/b7391860-4f37-11eb-8a36-ebb87efcf8c0.png" width="24%" />

::: warning 注意事项
- history 模式发行需要后台配置支持，详见：[history模式的后端配置](https://router.vuejs.org/zh/guide/essentials/history-mode.html#%E5%90%8E%E7%AB%AF%E9%85%8D%E7%BD%AE%E4%BE%8B%E5%AD%90) 
- 打包部署后，在服务器上开启 gzip 可以进一步压缩文件。具体的配置，可以参考网上的分享：[https://juejin.im/post/5af003286fb9a07aac24611b](https://juejin.im/post/5af003286fb9a07aac24611b)
:::

- **打包为微信小程序：** 

1. 申请微信小程序AppID，参考：[微信教程](https://developers.weixin.qq.com/miniprogram/dev/framework/quickstart/getstart.html#%E7%94%B3%E8%AF%B7%E5%B8%90%E5%8F%B7)
2. 在HBuilderX中顶部菜单依次点击 "发行" => "小程序-微信"，输入小程序名称和appid点击发行即可在 **`unpackage/dist/build/mp-weixin`** 生成微信小程序项目代码

<img src="https://bjetxgzv.cdn.bspapp.com/VKCEYUGU-uni-app-doc/b36294f0-4f37-11eb-8a36-ebb87efcf8c0.png" width="50%" />

3. 在微信小程序开发者工具中，导入生成的微信小程序项目，测试项目代码运行正常后，点击“上传”按钮，之后按照 “提交审核” => “发布” 小程序标准流程，逐步操作即可，详细查看：[微信官方教程](https://developers.weixin.qq.com/miniprogram/dev/framework/quickstart/release.html#%E5%8F%91%E5%B8%83%E4%B8%8A%E7%BA%BF)。

- 在 **`百度开发者工具`、`字节跳动开发者工具`、`快应用联盟`、`华为开发者工具`、`QQ小程序`** 发布：内容基本同上，不再重复。


## 通过vue-cli命令行

**除了HBuilderX可视化界面，也可以使用 `cli` 脚手架，可以通过 `vue-cli` 创建 `uni-app` 项目**

### 环境安装

全局安装vue-cli
```
npm install -g @vue/cli
```

### 运行、发布uni-app

```
npm run dev:%PLATFORM%
npm run build:%PLATFORM%
```
`%PLATFORM%` 可取值如下：
| **值** | **平台** |
| :-: | :-: |
| app-plus | app平台生成打包资源（支持npm run build:app-plus，可用于持续集成。不支持run，运行调试仍需在HBuilderX中操作） | 
| h5 | H5 | 
| mp-alipay | 支付宝小程序 | 
| mp-baidu | 百度小程序 | 
| mp-weixin | 微信小程序 | 
| mp-toutiao | 字节跳动小程序 | 
| mp-qq | QQ 小程序 | 
| mp-360 | 360 小程序 | 
| mp-kuaishou | 快手小程序 | 
| quickapp-webview | 快应用(webview) | 
| quickapp-webview-union | 快应用联盟 | 
| quickapp-webview-huawei | 快应用华为 | 

可以自定义更多条件编译平台，比如钉钉小程序，参考[package.json文档](https://uniapp.dcloud.io/collocation/package)。

#### 运行并发布快应用：
快应用有两种开发方式，uni-app均支持：
- 类小程序webview渲染方式：[https://ask.dcloud.net.cn/article/37182](https://ask.dcloud.net.cn/article/37182)
- 原生渲染方式：[https://ask.dcloud.net.cn/article/37145](https://ask.dcloud.net.cn/article/37145)

::: warning 注意事项
- 目前使用`npm run build:app-plus`会在 `/dist/build/app-plus` 下生成app打包资源。如需制作wgt包，将 `app-plus` 中的文件压缩成zip（注意：不要包含 `app-plus目录` ），再重命名为 `${appid}.wgt` ， `appid` 为 `manifest.json` 文件中的 `appid`。
- dev 模式编译出的各平台代码存放于根目录下的 `/dist/dev/` 目录，打开各平台开发工具选择对应平台目录即可进行预览（h5 平台不会在此目录，存在于缓存中）；
- build 模式编译出的各平台代码存放于根目录下的 `/dist/build/` 目录，发布时选择此目录进行发布；
- dev 和 build 模式的区别：

	1.dev 模式有 SourceMap 可以方便的进行断点调试；
	
	2.build 模式会将代码进行压缩，体积更小更适合发布为正式版应用；
	
	3.进行 环境判断 时，dev 模式 process.env.NODE_ENV 的值为 development，build 模式 process.env.NODE_ENV 的值为 production。
:::