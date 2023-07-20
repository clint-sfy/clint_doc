import { UserConfig } from 'vite';
import WindiCSS from 'vite-plugin-windicss';
import Icons, { ViteIconsResolver } from 'vite-plugin-icons';
import Components from 'vite-plugin-components';

const config: UserConfig = {
  server: {
    port: 3003, // 设置打开的端口号
  },
  optimizeDeps: {
    exclude: ['vue-demi', '@vueuse/shared', '@vueuse/core'],
  },
  plugins: [
    Components({
      dirs: ['.vitepress/theme/components'],
      customLoaderMatcher: (id) => id.endsWith('.md'),
      customComponentResolvers: [
        ViteIconsResolver({
          componentPrefix: '',
        }),
      ],
    }),
    Icons(),
    WindiCSS({
      preflight: false,
    }),
  ],
};

export default config;
