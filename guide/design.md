# 样式

## 介绍

主要介绍如何在项目中使用和规划样式文件。

默认使用 less 作为预处理语言，建议在使用前或者遇到疑问时学习一下 [Less](http://lesscss.org/) 的相关特性（如果想获取基础的 CSS 知识或查阅属性，请参考 [MDN 文档](https://developer.mozilla.org/zh-CN/docs/Web/CSS/Reference)）。

项目中使用的uniapp通用样式，都存放于 [uni.scss](https://gitee.com/kevin_chou/qdpz/blob/develop/uni.scss) 中。

#### 以下`ColorUi`,`uView`,`uni.scss` 插件的样式已在 App.vue/main.js 中全局引入，您可以直接使用
```bash
.
├── colorui 	# colorUi组件
├── uview-ui 	# uView组件
└── uni.scss 	# uniapp自带
```


## 使用colorUi

项目中引用到了 [colorUi](http://docs.xzeu.com/#/),具体可以见文件使用说明。

语法如下:

```html
<div class="padding-right-lg bg-gray"></div>
```

## 使用uView

项目中引用到了 [uView](https://www.uviewui.com/)，具体可以见文件使用说明。

我们在全局样式中，通过scss提供了对应的颜色变量名，方便您在任何可写css的地方，调用这些变量，如下：

```css
/* 变量的定义，该部分uView已全局引入，无需您编写 */
$u-type-primary: #2979ff;
$u-type-primary-light: #ecf5ff;
$u-type-primary-disabled: #a0cfff;
$u-type-primary-dark: #2b85e4;


/* 在您编写css的地方使用这些变量 */
.title {
	color: $u-type-primary;
	......
}
```

## 使用uni.scss
::: tip 温馨提醒

uni.scss文件的用途是为了方便整体控制应用的风格。比如按钮颜色、边框风格，uni.scss文件里预置了一批scss变量预置。

uni.scss 这个文件会被全局注入到所有文件，所以在页面内可以直接使用变量而不需要手动引入

uni-app 官方扩展插件及插件市场（https://ext.dcloud.net.cn）上很多三方插件均使用了这些样式变量

如果你是插件开发者，建议你使用scss预处理，并在插件代码中直接使用这些变量（无需 import 这个文件），方便用户通过搭积木的方式开发整体风格一致的App

如果你是App开发者（插件使用者），你可以通过修改这些变量来定制自己的插件主题，实现自定义主题功能

:::

#### `注意`
- 如要使用这些常用变量，需要在 HBuilderX 里面安装 scss 插件；
- 使用时需要在 style 节点上加上 lang="scss"。
```html
<style lang="scss">
......
</style>
```
- pages.json不支持scss，原生导航栏和tabbar的动态修改只能使用js api

**以下是 uni.scss 的相关变量：**

```css
/* 颜色变量 */

/* 行为相关颜色 */
$uni-color-primary: #007aff;
$uni-color-success: #4cd964;
$uni-color-warning: #f0ad4e;
$uni-color-error: #dd524d;

/* 文字基本颜色 */
$uni-text-color:#333;//基本色
$uni-text-color-inverse:#fff;//反色
$uni-text-color-grey:#999;//辅助灰色，如加载更多的提示信息
$uni-text-color-placeholder: #808080;
$uni-text-color-disable:#c0c0c0;

/* 背景颜色 */
$uni-bg-color:#ffffff;
$uni-bg-color-grey:#f8f8f8;
$uni-bg-color-hover:#f1f1f1;//点击状态颜色
$uni-bg-color-mask:rgba(0, 0, 0, 0.4);//遮罩颜色

/* 边框颜色 */
$uni-border-color:#c8c7cc;

/* 尺寸变量 */

/* 文字尺寸 */
$uni-font-size-sm:24rpx;
$uni-font-size-base:28rpx;
$uni-font-size-lg:32rpx;

/* 图片尺寸 */
$uni-img-size-sm:40rpx;
$uni-img-size-base:52rpx;
$uni-img-size-lg:80rpx;

/* Border Radius */
$uni-border-radius-sm: 4rpx;
$uni-border-radius-base: 6rpx;
$uni-border-radius-lg: 12rpx;
$uni-border-radius-circle: 50%;

/* 水平间距 */
$uni-spacing-row-sm: 10px;
$uni-spacing-row-base: 20rpx;
$uni-spacing-row-lg: 30rpx;

/* 垂直间距 */
$uni-spacing-col-sm: 8rpx;
$uni-spacing-col-base: 16rpx;
$uni-spacing-col-lg: 24rpx;

/* 透明度 */
$uni-opacity-disabled: 0.3; // 组件禁用态的透明度

/* 文章场景相关 */
$uni-color-title: #2C405A; // 文章标题颜色
$uni-font-size-title:40rpx;
$uni-color-subtitle: #555555; // 二级标题颜色
$uni-font-size-subtitle:36rpx;
$uni-color-paragraph: #3F536E; // 文章段落颜色
$uni-font-size-paragraph:30rpx;
```


## 开启 scoped

没有加 `scoped` 属性，默认会编译成全局样式，可能会造成全局污染

```vue
<style></style>

<style scoped></style>
```

::: tip 温馨提醒

使用 scoped 后，父组件的样式将不会渗透到子组件中。不过一个子组件的根节点会同时受其父组件的 scoped CSS 和子组件的 scoped CSS 的影响。这样设计是为了让父组件可以从布局的角度出发，调整其子组件根元素的样式。

:::

## 深度选择器

有时我们可能想明确地制定一个针对子组件的规则。

如果你希望 `scoped` 样式中的一个选择器能够作用得“更深”，例如影响子组件，你可以使用 `>>>` 操作符。有些像 Sass 之类的预处理器无法正确解析 `>>>`。这种情况下你可以使用 `/deep/` 或 `::v-deep` 操作符取而代之——两者都是 `>>>` 的别名，同样可以正常工作。

详情可以查看 RFC[0023-scoped-styles-changes](https://github.com/vuejs/rfcs/blob/master/active-rfcs/0023-scoped-styles-changes.md)。

使用 scoped 后，父组件的样式将不会渗透到子组件中
