import{f as a,u as t,g as e,h as s,i as l,o as i,c as o,d as c,b as r,t as n,_ as v,p as d,a as u,j as f,F as m,r as p,w as h,k as g,l as k}from"./app.a88d1aa8.js";d("data-v-13006a62");const _={key:0,class:"home-hero"},y={key:0,class:"figure"},x={key:1,id:"main-title",class:"title"},$=u('<div class="gitBox" data-v-13006a62><a href="https://gitee.com/clint_sfy/clint_doc" target="_blank" data-v-13006a62><img src="https://svg.hamm.cn/gitee.svg?type=star&amp;user=clint_sfy&amp;project=clint_doc" data-v-13006a62></a><a href="https://gitee.com/clint_sfy/clint_doc/members" target="_blank" data-v-13006a62><img src="https://svg.hamm.cn/gitee.svg?type=fork&amp;user=clint_sfy&amp;project=clint_doc" data-v-13006a62></a></div>',1),b={key:2,class:"description"};f();var I=a({expose:[],setup(a){const d=t(),u=e(),f=s((()=>u.value.heroImage||m.value||h.value||k.value)),m=s((()=>null!==u.value.heroText)),p=s((()=>u.value.heroText||d.value.title)),h=s((()=>null!==u.value.tagline)),g=s((()=>u.value.tagline||d.value.description)),k=s((()=>u.value.actionLink&&u.value.actionText)),I=s((()=>u.value.altActionLink&&u.value.altActionText));return(a,t)=>l(f)?(i(),o("header",_,[a.$frontmatter.heroImage?(i(),o("figure",y,[c("img",{class:"image",src:a.$withBase(a.$frontmatter.heroImage),alt:a.$frontmatter.heroAlt},null,8,["src","alt"])])):r("v-if",!0),l(m)?(i(),o("h1",x,n(l(p)),1)):r("v-if",!0),$,l(h)?(i(),o("p",b,n(l(g)),1)):r("v-if",!0),l(k)?(i(),o(v,{key:3,item:{link:l(u).actionLink,text:l(u).actionText},class:"action"},null,8,["item"])):r("v-if",!0),l(I)?(i(),o(v,{key:4,item:{link:l(u).altActionLink,text:l(u).altActionText},class:"action alt"},null,8,["item"])):r("v-if",!0)])):r("v-if",!0)}});I.__scopeId="data-v-13006a62",d("data-v-608136a8");const T={key:0,class:"home-features"},A={class:"wrapper"},j={class:"container"},L={class:"features"},w={key:0,class:"title"},B={key:1,class:"details"};f();var C=a({expose:[],setup(a){const t=e(),v=s((()=>t.value.features&&t.value.features.length>0)),d=s((()=>t.value.features?t.value.features:[]));return(a,t)=>l(v)?(i(),o("div",T,[c("div",A,[c("div",j,[c("div",L,[(i(!0),o(m,null,p(l(d),((a,t)=>(i(),o("section",{key:t,class:"feature"},[a.title?(i(),o("h2",w,n(a.title),1)):r("v-if",!0),a.details?(i(),o("p",B,n(a.details),1)):r("v-if",!0)])))),128))])])])])):r("v-if",!0)}});C.__scopeId="data-v-608136a8";const F={},q=h();d("data-v-592eb986");const z={key:0,class:"footer"},D={class:"container"},E={class:"text"};f();const G=q(((a,t)=>a.$frontmatter.footer?(i(),o("footer",z,[c("div",D,[c("p",E,n(a.$frontmatter.footer),1)])])):r("v-if",!0)));F.render=G,F.__scopeId="data-v-592eb986",d("data-v-9d516c10");const H={class:"home","aria-labelledby":"main-title"},J={class:"home-content"};f();var K=a({expose:[],setup:a=>(a,t)=>{const e=g("Content");return i(),o("main",H,[c(I),k(a.$slots,"hero",{},void 0,!0),c(C),c("div",J,[c(e)]),k(a.$slots,"features",{},void 0,!0),c(F),k(a.$slots,"footer",{},void 0,!0)])}});K.__scopeId="data-v-9d516c10";export default K;
