<template>
  <a
    class="nav-bar-title"
    :href="$withBase($localePath)"
    :aria-label="`${$siteByRoute.title}, back to home`"
  >
    <img
      v-if="$themeConfig.logo"
      class="logo"
      :src="$withBase($themeConfig.logo)"
      alt="Logo"
    />
    <!-- {{ $site.title }} -->
	<text style="margin-top: -1px;">阿源编程 | 官方文档</text>
  </a>
</template>

<style scoped>
.nav-bar-title {
  min-width: 12rem;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--c-text);
}

.nav-bar-title:hover {
  text-decoration: none;
}

.logo {
  margin-right: 0.7rem;
  height: 1.3rem;
  vertical-align: bottom;
}
</style>
