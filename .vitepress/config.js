// @ts-check
/**
 * @type {import('vitepress').UserConfig}
 */
module.exports = {
  title: '阿源编程',
  lang: 'zh-CN',
  description: '一个用于展示项目文档和个人笔记的web网站',
  head: createHead(),
  themeConfig: {
    repo: 'jekip/naive-ui-admin',
    docsRepo: 'jekip/naive-ui-admin-docs',
    logo: '/logo64.png',
    docsBranch: 'main',
    editLinks: true,
    editLinkText: '为此页提供修改建议',
    nav: createNav(),
    sidebar: createSidebar(),
  },
};

/**
 * @type {()=>import('vitepress').HeadConfig[]}
 */

function createHead() {
  return [
    ['meta', { name: 'author', content: 'Vbenjs Team' }],
    [
      'meta',
      {
        name: 'keywords',
        content: 'vben, vitejs, vite, ant-design-vue, vue',
      },
    ],
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    [
      'meta',
      {
        name: 'viewport',
        content:
          'width=device-width,initial-scale=1,minimum-scale=1.0,maximum-scale=1.0,user-scalable=no',
      },
    ],
    ['meta', { name: 'keywords', content: 'vue vben admin docs' }],
    ['link', { rel: 'icon', href: '/favicon32.ico' }],
  ];
}

/**
 * @type {()=>import('./theme-default/config').DefaultTheme.NavItem[]}
 */
function createNav() {
  return [
    {
      text: '首页',
      link: '/guide/introduction'
    },
    {
      text: '项目文档',
      link: '/project/index',
    },
    {
      text: '好玩的',
      items:[
        {
          text: '开源项目',
          link: '',
        },
        {
          text: 'ChatGpt',
          link: '/intresting/ChatGPT',
        },
        {
          text: 'AI行业对话',
          link: '/intresting/AI_talk',
        },
      ]
    },
    {
      text: '科研笔记',
      items:[
        {
          text: 'python基础',
          link: '/note/python/01_python_base',
        },
        {
          text: '数学基础',
          link: '/note/math/02_math_base',
        },
        {
          text: '机器学习',
          link: '/note/machine_learning/03_machine_base',
        },
        {
          text: '深度学习',
          link: '/note/deep_learning/3_pytorh',
        },
      ]

    },
    {
      text: '精读论文',
      items:[
        {
          text: 'Cv',
          link: '',
        },
        {
          text: 'NLP',
          link: '',
        },
      ]
    },
    {
      text: '相关链接',
      items: [
        {
          text: '在线预览',
          link: '/other/see',
        },
        {
          text: '项目源码',
          link: 'https://gitee.com/clint_sfy/clint_doc',
        },
        {
          text: '文档源码',
          link: 'https://gitee.com/clint_sfy/clint_doc',
        },
		{
		  text: '点赞支持',
		  link: 'https://gitee.com/clint_sfy/clint_doc/stargazers',
		}
      ],
    },
    {
      text: '赞助',
      link: '/other/donate',
    },
  ];
}

function createSidebar() {
  return {
    '/note/math/':[
      {
        text: 'math基础',
        children: [
          {
            text: 'math基础',
            link: '/note/math/02_math_base',
          },
        ],       
      }
    ],
    '/note/python/':[
      {
        text: 'python基础',
        children: [
          {
            text: 'python',
            link: '/note/python/01_python_base',
          },
        ],       
      }
    ],
    '/note/machine_learning/':[
      {
        text: '机器学习',
        children: [
          {
            text: '机器学习基础',
            link: '/note/machine_learning/03_machine_base',
          },
        ],       
      }
    ],
    '/note/deep_learning/':[
      {
        text: '深度学习',
        children: [
          {
            text: '1_pytorch',
            link: '/note/deep_learning/3_pytorh',
          },
          {
            text: '2_mmlab',
            link: '/note/deep_learning/4_MMLab',
          }
        ],       
      }
    ],
    '/intresting/': [
      {
        text: '有趣的ChatGPT',
        children: [
          {
            text: 'ChatGPT',
            link: '/intresting/ChatGPT',
          }
        ],       
      },
      {
        text: '有趣的AI',
        children: [
          {
            text: 'AI行业对话',
            link: '/intresting/AI_talk',
          }
        ],       
      },
    ],
    '/': [
      {
        text: '首页',
        children: [
          {
            text: '介绍',
            link: '/guide/introduction',
          },
          {
            text: '开始',
            link: '/guide/',
          },
        ],
      }
    ],
  };
}
