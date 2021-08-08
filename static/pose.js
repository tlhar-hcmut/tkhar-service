(function(){/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
'use strict';var B;function aa(a){var c=0;return function(){return c<a.length?{done:!1,value:a[c++]}:{done:!0}}}var C="function"==typeof Object.defineProperties?Object.defineProperty:function(a,c,b){if(a==Array.prototype||a==Object.prototype)return a;a[c]=b.value;return a};
function ba(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var c=0;c<a.length;++c){var b=a[c];if(b&&b.Math==Math)return b}throw Error("Cannot find global object");}var D=ba(this);function G(a,c){if(c)a:{var b=D;a=a.split(".");for(var d=0;d<a.length-1;d++){var g=a[d];if(!(g in b))break a;b=b[g]}a=a[a.length-1];d=b[a];c=c(d);c!=d&&null!=c&&C(b,a,{configurable:!0,writable:!0,value:c})}}
G("Symbol",function(a){function c(h){if(this instanceof c)throw new TypeError("Symbol is not a constructor");return new b(d+(h||"")+"_"+g++,h)}function b(h,e){this.g=h;C(this,"description",{configurable:!0,writable:!0,value:e})}if(a)return a;b.prototype.toString=function(){return this.g};var d="jscomp_symbol_"+(1E9*Math.random()>>>0)+"_",g=0;return c});
G("Symbol.iterator",function(a){if(a)return a;a=Symbol("Symbol.iterator");for(var c="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),b=0;b<c.length;b++){var d=D[c[b]];"function"===typeof d&&"function"!=typeof d.prototype[a]&&C(d.prototype,a,{configurable:!0,writable:!0,value:function(){return ca(aa(this))}})}return a});function ca(a){a={next:a};a[Symbol.iterator]=function(){return this};return a}
function I(a){var c="undefined"!=typeof Symbol&&Symbol.iterator&&a[Symbol.iterator];return c?c.call(a):{next:aa(a)}}function fa(a){if(!(a instanceof Array)){a=I(a);for(var c,b=[];!(c=a.next()).done;)b.push(c.value);a=b}return a}var K;if("function"==typeof Object.setPrototypeOf)K=Object.setPrototypeOf;else{var L;a:{var ha={a:!0},ia={};try{ia.__proto__=ha;L=ia.a;break a}catch(a){}L=!1}K=L?function(a,c){a.__proto__=c;if(a.__proto__!==c)throw new TypeError(a+" is not extensible");return a}:null}
var ja=K;function Q(){this.l=!1;this.i=null;this.h=void 0;this.g=1;this.s=this.m=0;this.j=null}function R(a){if(a.l)throw new TypeError("Generator is already running");a.l=!0}Q.prototype.o=function(a){this.h=a};function S(a,c){a.j={S:c,T:!0};a.g=a.m||a.s}Q.prototype.return=function(a){this.j={return:a};this.g=this.s};function T(a,c,b){a.g=b;return{value:c}}function ka(a){this.g=new Q;this.h=a}
function la(a,c){R(a.g);var b=a.g.i;if(b)return U(a,"return"in b?b["return"]:function(d){return{value:d,done:!0}},c,a.g.return);a.g.return(c);return V(a)}function U(a,c,b,d){try{var g=c.call(a.g.i,b);if(!(g instanceof Object))throw new TypeError("Iterator result "+g+" is not an object");if(!g.done)return a.g.l=!1,g;var h=g.value}catch(e){return a.g.i=null,S(a.g,e),V(a)}a.g.i=null;d.call(a.g,h);return V(a)}
function V(a){for(;a.g.g;)try{var c=a.h(a.g);if(c)return a.g.l=!1,{value:c.value,done:!1}}catch(b){a.g.h=void 0,S(a.g,b)}a.g.l=!1;if(a.g.j){c=a.g.j;a.g.j=null;if(c.T)throw c.S;return{value:c.return,done:!0}}return{value:void 0,done:!0}}
function ma(a){this.next=function(c){R(a.g);a.g.i?c=U(a,a.g.i.next,c,a.g.o):(a.g.o(c),c=V(a));return c};this.throw=function(c){R(a.g);a.g.i?c=U(a,a.g.i["throw"],c,a.g.o):(S(a.g,c),c=V(a));return c};this.return=function(c){return la(a,c)};this[Symbol.iterator]=function(){return this}}function W(a,c){c=new ma(new ka(c));ja&&a.prototype&&ja(c,a.prototype);return c}
var na="function"==typeof Object.assign?Object.assign:function(a,c){for(var b=1;b<arguments.length;b++){var d=arguments[b];if(d)for(var g in d)Object.prototype.hasOwnProperty.call(d,g)&&(a[g]=d[g])}return a};G("Object.assign",function(a){return a||na});
G("Promise",function(a){function c(e){this.h=0;this.i=void 0;this.g=[];this.o=!1;var f=this.j();try{e(f.resolve,f.reject)}catch(k){f.reject(k)}}function b(){this.g=null}function d(e){return e instanceof c?e:new c(function(f){f(e)})}if(a)return a;b.prototype.h=function(e){if(null==this.g){this.g=[];var f=this;this.i(function(){f.l()})}this.g.push(e)};var g=D.setTimeout;b.prototype.i=function(e){g(e,0)};b.prototype.l=function(){for(;this.g&&this.g.length;){var e=this.g;this.g=[];for(var f=0;f<e.length;++f){var k=
e[f];e[f]=null;try{k()}catch(m){this.j(m)}}}this.g=null};b.prototype.j=function(e){this.i(function(){throw e;})};c.prototype.j=function(){function e(m){return function(n){k||(k=!0,m.call(f,n))}}var f=this,k=!1;return{resolve:e(this.B),reject:e(this.l)}};c.prototype.B=function(e){if(e===this)this.l(new TypeError("A Promise cannot resolve to itself"));else if(e instanceof c)this.D(e);else{a:switch(typeof e){case "object":var f=null!=e;break a;case "function":f=!0;break a;default:f=!1}f?this.A(e):this.m(e)}};
c.prototype.A=function(e){var f=void 0;try{f=e.then}catch(k){this.l(k);return}"function"==typeof f?this.F(f,e):this.m(e)};c.prototype.l=function(e){this.s(2,e)};c.prototype.m=function(e){this.s(1,e)};c.prototype.s=function(e,f){if(0!=this.h)throw Error("Cannot settle("+e+", "+f+"): Promise already settled in state"+this.h);this.h=e;this.i=f;2===this.h&&this.C();this.u()};c.prototype.C=function(){var e=this;g(function(){if(e.v()){var f=D.console;"undefined"!==typeof f&&f.error(e.i)}},1)};c.prototype.v=
function(){if(this.o)return!1;var e=D.CustomEvent,f=D.Event,k=D.dispatchEvent;if("undefined"===typeof k)return!0;"function"===typeof e?e=new e("unhandledrejection",{cancelable:!0}):"function"===typeof f?e=new f("unhandledrejection",{cancelable:!0}):(e=D.document.createEvent("CustomEvent"),e.initCustomEvent("unhandledrejection",!1,!0,e));e.promise=this;e.reason=this.i;return k(e)};c.prototype.u=function(){if(null!=this.g){for(var e=0;e<this.g.length;++e)h.h(this.g[e]);this.g=null}};var h=new b;c.prototype.D=
function(e){var f=this.j();e.I(f.resolve,f.reject)};c.prototype.F=function(e,f){var k=this.j();try{e.call(f,k.resolve,k.reject)}catch(m){k.reject(m)}};c.prototype.then=function(e,f){function k(r,p){return"function"==typeof r?function(u){try{m(r(u))}catch(l){n(l)}}:p}var m,n,t=new c(function(r,p){m=r;n=p});this.I(k(e,m),k(f,n));return t};c.prototype.catch=function(e){return this.then(void 0,e)};c.prototype.I=function(e,f){function k(){switch(m.h){case 1:e(m.i);break;case 2:f(m.i);break;default:throw Error("Unexpected state: "+
m.h);}}var m=this;null==this.g?h.h(k):this.g.push(k);this.o=!0};c.resolve=d;c.reject=function(e){return new c(function(f,k){k(e)})};c.race=function(e){return new c(function(f,k){for(var m=I(e),n=m.next();!n.done;n=m.next())d(n.value).I(f,k)})};c.all=function(e){var f=I(e),k=f.next();return k.done?d([]):new c(function(m,n){function t(u){return function(l){r[u]=l;p--;0==p&&m(r)}}var r=[],p=0;do r.push(void 0),p++,d(k.value).I(t(r.length-1),n),k=f.next();while(!k.done)})};return c});
function oa(a,c){a instanceof String&&(a+="");var b=0,d=!1,g={next:function(){if(!d&&b<a.length){var h=b++;return{value:c(h,a[h]),done:!1}}d=!0;return{done:!0,value:void 0}}};g[Symbol.iterator]=function(){return g};return g}G("Array.prototype.keys",function(a){return a?a:function(){return oa(this,function(c){return c})}});var sa=this||self;
function X(a,c){a=a.split(".");var b=sa;a[0]in b||"undefined"==typeof b.execScript||b.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)a.length||void 0===c?b[d]&&b[d]!==Object.prototype[d]?b=b[d]:b=b[d]={}:b[d]=c};function Y(a,c){var b=void 0;return new (b||(b=Promise))(function(d,g){function h(k){try{f(c.next(k))}catch(m){g(m)}}function e(k){try{f(c["throw"](k))}catch(m){g(m)}}function f(k){k.done?d(k.value):(new b(function(m){m(k.value)})).then(h,e)}f((c=c.apply(a,void 0)).next())})};function ta(a,c,b){b=a.createShader(0===b?a.VERTEX_SHADER:a.FRAGMENT_SHADER);a.shaderSource(b,c);a.compileShader(b);if(!a.getShaderParameter(b,a.COMPILE_STATUS))throw Error("Could not compile WebGL shader.\n\n"+a.getShaderInfoLog(b));return b};function ua(a,c,b){this.h=a;this.g=c;this.u=b;this.l=0}function va(a){if("function"===typeof a.g.canvas.transferToImageBitmap)return Promise.resolve(a.g.canvas.transferToImageBitmap());if(a.u)return Promise.resolve(a.g.canvas);if("function"===typeof createImageBitmap)return createImageBitmap(a.g.canvas);void 0===a.j&&(a.j=document.createElement("img"));return new Promise(function(c){a.j.onload=function(){requestAnimationFrame(function(){c(a.j)})};a.j.src=a.g.canvas.toDataURL()})}
function wa(a,c){var b=a.g;if(void 0===a.m){var d=ta(b,"\n  attribute vec2 aVertex;\n  attribute vec2 aTex;\n  varying vec2 vTex;\n  void main(void) {\n    gl_Position = vec4(aVertex, 0.0, 1.0);\n    vTex = aTex;\n  }",0),g=ta(b,"\n  precision mediump float;\n  varying vec2 vTex;\n  uniform sampler2D sampler0;\n  void main(){\n    gl_FragColor = texture2D(sampler0, vTex);\n  }",1),h=b.createProgram();b.attachShader(h,d);b.attachShader(h,g);b.linkProgram(h);if(!b.getProgramParameter(h,b.LINK_STATUS))throw Error("Could not compile WebGL program.\n\n"+
b.getProgramInfoLog(h));d=a.m=h;b.useProgram(d);g=b.getUniformLocation(d,"sampler0");a.i={H:b.getAttribLocation(d,"aVertex"),G:b.getAttribLocation(d,"aTex"),V:g};a.s=b.createBuffer();b.bindBuffer(b.ARRAY_BUFFER,a.s);b.enableVertexAttribArray(a.i.H);b.vertexAttribPointer(a.i.H,2,b.FLOAT,!1,0,0);b.bufferData(b.ARRAY_BUFFER,new Float32Array([-1,-1,-1,1,1,1,1,-1]),b.STATIC_DRAW);b.bindBuffer(b.ARRAY_BUFFER,null);a.o=b.createBuffer();b.bindBuffer(b.ARRAY_BUFFER,a.o);b.enableVertexAttribArray(a.i.G);b.vertexAttribPointer(a.i.G,
2,b.FLOAT,!1,0,0);b.bufferData(b.ARRAY_BUFFER,new Float32Array([0,1,0,0,1,0,1,1]),b.STATIC_DRAW);b.bindBuffer(b.ARRAY_BUFFER,null);b.uniform1i(g,0)}d=a.i;b.useProgram(a.m);b.canvas.width=c.width;b.canvas.height=c.height;b.viewport(0,0,c.width,c.height);b.activeTexture(b.TEXTURE0);a.h.bindTexture2d(c.glName);b.enableVertexAttribArray(d.H);b.bindBuffer(b.ARRAY_BUFFER,a.s);b.vertexAttribPointer(d.H,2,b.FLOAT,!1,0,0);b.enableVertexAttribArray(d.G);b.bindBuffer(b.ARRAY_BUFFER,a.o);b.vertexAttribPointer(d.G,
2,b.FLOAT,!1,0,0);b.bindFramebuffer(b.DRAW_FRAMEBUFFER?b.DRAW_FRAMEBUFFER:b.FRAMEBUFFER,null);b.clearColor(0,0,0,0);b.clear(b.COLOR_BUFFER_BIT);b.colorMask(!0,!0,!0,!0);b.drawArrays(b.TRIANGLE_FAN,0,4);b.disableVertexAttribArray(d.H);b.disableVertexAttribArray(d.G);b.bindBuffer(b.ARRAY_BUFFER,null);a.h.bindTexture2d(0)}function xa(a){this.g=a};var ya=new Uint8Array([0,97,115,109,1,0,0,0,1,4,1,96,0,0,3,2,1,0,10,9,1,7,0,65,0,253,15,26,11]);function za(a,c){return c+a}function Aa(a,c){window[a]=c}function Ba(a){var c=document.createElement("script");c.setAttribute("src",a);c.setAttribute("crossorigin","anonymous");document.body.appendChild(c);return new Promise(function(b){c.addEventListener("load",function(){b()},!1)})}
function Ca(a){for(var c=[],b=a.size(),d=0;d<b;++d){var g=a.get(d);c.push({x:g.x,y:g.y,z:g.z,visibility:g.hasVisibility?g.visibility:void 0});g.delete()}return c}function Da(a){for(var c=[],b=a.size(),d=0;d<b;++d){var g=a.get(d);c.push({index:g.index,score:g.score,label:g.hasLabel?g.label:void 0,displayName:g.hasDisplayName?g.displayName:void 0})}return c}
function Ea(){return Y(this,function c(){return W(c,function(b){switch(b.g){case 1:return b.m=2,T(b,WebAssembly.instantiate(ya),4);case 4:b.g=3;b.m=0;break;case 2:return b.m=0,b.j=null,b.return(!1);case 3:return b.return(!0)}})})}
function Fa(a){this.g=a;this.listeners={};this.l={};this.B={};this.m={};this.s={};this.v=this.o=this.O=!0;this.F=Promise.resolve();this.N="";this.u={};this.locateFile=a&&a.locateFile||za;if("object"===typeof window)a=window.location.pathname.toString().substring(0,window.location.pathname.toString().lastIndexOf("/"))+"/";else if("undefined"!==typeof location)a=location.pathname.toString().substring(0,location.pathname.toString().lastIndexOf("/"))+"/";else throw Error("solutions can only be loaded on a web page or in a web worker");
this.P=a}B=Fa.prototype;B.close=function(){this.i&&this.i.delete();return Promise.resolve()};function Ga(a,c){return void 0===a.g.files?[]:"function"===typeof a.g.files?a.g.files(c):a.g.files}
function Ha(a){return Y(a,function b(){var d=this,g,h,e,f,k,m,n,t,r,p,u;return W(b,function(l){switch(l.g){case 1:g=d;if(!d.O)return l.return();h=Ga(d,d.l);return T(l,Ea(),2);case 2:e=l.h;if("object"===typeof window)return Aa("createMediapipeSolutionsWasm",{locateFile:d.locateFile}),Aa("createMediapipeSolutionsPackedAssets",{locateFile:d.locateFile}),m=h.filter(function(q){return void 0!==q.data}),n=h.filter(function(q){return void 0===q.data}),t=Promise.all(m.map(function(q){return Z(g,q.url)})),
r=Promise.all(n.map(function(q){return void 0===q.simd||q.simd&&e||!q.simd&&!e?Ba(g.locateFile(q.url,g.P)):Promise.resolve()})).then(function(){return Y(g,function v(){var w,y,z=this;return W(v,function(A){if(1==A.g)return w=window.createMediapipeSolutionsWasm,y=window.createMediapipeSolutionsPackedAssets,T(A,w(y),2);z.h=A.h;A.g=0})})}),p=function(){return Y(g,function v(){var w=this;return W(v,function(y){w.g.graph&&w.g.graph.url?y=T(y,Z(w,w.g.graph.url),0):(y.g=0,y=void 0);return y})})}(),T(l,Promise.all([r,
t,p]),7);if("function"!==typeof importScripts)throw Error("solutions can only be loaded on a web page or in a web worker");f=h.filter(function(q){return void 0===q.simd||q.simd&&e||!q.simd&&!e}).map(function(q){return g.locateFile(q.url,g.P)});importScripts.apply(null,fa(f));return T(l,createMediapipeSolutionsWasm(Module),6);case 6:d.h=l.h;d.j=new OffscreenCanvas(1,1);d.h.canvas=d.j;k=d.h.GL.createContext(d.j,{antialias:!1,alpha:!1,U:"undefined"!==typeof WebGL2RenderingContext?2:1});d.h.GL.makeContextCurrent(k);
l.g=4;break;case 7:d.j=document.createElement("canvas");u=d.j.getContext("webgl2",{});if(!u&&(u=d.j.getContext("webgl",{}),!u))return alert("Failed to create WebGL canvas context when passing video frame."),l.return();d.A=u;d.h.canvas=d.j;d.h.createContext(d.j,!0,!0,{});case 4:d.i=new d.h.SolutionWasm,d.O=!1,l.g=0}})})}
function Ia(a){return Y(a,function b(){var d=this,g,h,e,f,k,m,n,t;return W(b,function(r){if(1==r.g){if(d.g.graph&&d.g.graph.url&&d.N===d.g.graph.url)return r.return();d.o=!0;if(!d.g.graph||!d.g.graph.url){r.g=2;return}d.N=d.g.graph.url;return T(r,Z(d,d.g.graph.url),3)}2!=r.g&&(g=r.h,d.i.loadGraph(g));h=I(Object.keys(d.u));for(e=h.next();!e.done;e=h.next())f=e.value,d.i.overrideFile(f,d.u[f]);d.u={};if(d.g.listeners)for(k=I(d.g.listeners),m=k.next();!m.done;m=k.next())n=m.value,Ja(d,n);t=d.l;d.l={};
d.setOptions(t);r.g=0})})}B.reset=function(){return Y(this,function c(){var b=this;return W(c,function(d){b.i&&(b.i.reset(),b.m={},b.s={});d.g=0})})};
B.setOptions=function(a){var c=this;if(this.g.options){for(var b=[],d=[],g={},h=I(Object.keys(a)),e=h.next();!e.done;g={J:g.J,K:g.K},e=h.next()){var f=e.value;!(f in this.l&&this.l[f]===a[f])&&(this.l[f]=a[f],e=this.g.options[f])&&(e.onChange&&(g.J=e.onChange,g.K=a[f],b.push(function(k){return function(){return Y(c,function n(){var t,r=this;return W(n,function(p){if(1==p.g)return T(p,k.J(k.K),2);t=p.h;!0===t&&(r.o=!0);p.g=0})})}}(g))),e.graphOptionXref&&(f={valueNumber:0===e.type?a[f]:0,valueBoolean:1===
e.type?a[f]:!1},e=Object.assign(Object.assign(Object.assign({},{calculatorName:"",calculatorIndex:0}),e.graphOptionXref),f),d.push(e)))}if(0!==b.length||0!==d.length)this.o=!0,this.C=d,this.D=b}};
function Ka(a){return Y(a,function b(){var d=this,g,h,e,f,k,m,n;return W(b,function(t){switch(t.g){case 1:if(!d.o)return t.return();if(!d.D){t.g=2;break}g=I(d.D);h=g.next();case 3:if(h.done){t.g=5;break}e=h.value;return T(t,e(),4);case 4:h=g.next();t.g=3;break;case 5:d.D=void 0;case 2:if(d.C){f=new d.h.GraphOptionChangeRequestList;k=I(d.C);for(m=k.next();!m.done;m=k.next())n=m.value,f.push_back(n);d.i.changeOptions(f);f.delete();d.C=void 0}d.o=!1;t.g=0}})})}
B.initialize=function(){return Y(this,function c(){var b=this;return W(c,function(d){return 1==d.g?T(d,Ha(b),2):3!=d.g?T(d,Ia(b),3):T(d,Ka(b),0)})})};function Z(a,c){return Y(a,function d(){var g=this,h,e;return W(d,function(f){if(c in g.B)return f.return(g.B[c]);h=g.locateFile(c,"");e=fetch(h).then(function(k){return k.arrayBuffer()});g.B[c]=e;return f.return(e)})})}B.overrideFile=function(a,c){this.i?this.i.overrideFile(a,c):this.u[a]=c};B.clearOverriddenFiles=function(){this.u={};this.i&&this.i.clearOverriddenFiles()};
B.send=function(a,c){return Y(this,function d(){var g=this,h,e,f,k,m,n,t,r,p;return W(d,function(u){if(1==u.g){if(!g.g.inputs)return u.return();h=1E3*(void 0===c||null===c?performance.now():c);return T(u,g.F,2)}if(3!=u.g)return T(u,g.initialize(),3);e=new g.h.PacketDataList;f=I(Object.keys(a));for(k=f.next();!k.done;k=f.next())if(m=k.value,n=g.g.inputs[m]){a:{var l=g;var q=a[m];switch(n.type){case "video":var v=l.m[n.stream];v||(v=new ua(l.h,l.A,l.v),l.m[n.stream]=v);l=v;0===l.l&&(l.l=l.h.createTexture());
if("undefined"!==typeof HTMLVideoElement&&q instanceof HTMLVideoElement){var w=q.videoWidth;v=q.videoHeight}else"undefined"!==typeof HTMLImageElement&&q instanceof HTMLImageElement?(w=q.naturalWidth,v=q.naturalHeight):(w=q.width,v=q.height);v={glName:l.l,width:w,height:v};w=l.g;w.canvas.width=v.width;w.canvas.height=v.height;w.activeTexture(w.TEXTURE0);l.h.bindTexture2d(l.l);w.texImage2D(w.TEXTURE_2D,0,w.RGBA,w.RGBA,w.UNSIGNED_BYTE,q);l.h.bindTexture2d(0);l=v;break a;case "detections":v=l.m[n.stream];
v||(v=new xa(l.h),l.m[n.stream]=v);l=v;l.data||(l.data=new l.g.DetectionListData);l.data.reset(q.length);for(v=0;v<q.length;++v){w=q[v];l.data.setBoundingBox(v,w.R);if(w.M)for(var y=0;y<w.M.length;++y){var z=w.M[y],A=z.visibility?!0:!1;l.data.addNormalizedLandmark(v,Object.assign(Object.assign({},z),{hasVisibility:A,visibility:A?z.visibility:0}))}if(w.L)for(y=0;y<w.L.length;++y){z=w.L[y];A=z.index?!0:!1;var O=z.label?!0:!1,M=z.displayName?!0:!1;l.data.addClassification(v,{score:z.score,hasIndex:A,
index:A?z.index:-1,hasLabel:O,label:O?z.label:"",hasDisplayName:M,displayName:M?z.displayName:""})}}l=l.data;break a;default:l={}}}t=l;r=n.stream;switch(n.type){case "video":e.pushTexture2d(Object.assign(Object.assign({},t),{stream:r,timestamp:h}));break;case "detections":p=t;p.stream=r;p.timestamp=h;e.pushDetectionList(p);break;default:throw Error("Unknown input config type: '"+n.type+"'");}}g.i.send(e);e.delete();u.g=0})})};
function La(a,c,b){return Y(a,function g(){var h,e,f,k,m,n,t=this,r,p,u,l,q,v,w,y,z,A,O,M,pa;return W(g,function(E){switch(E.g){case 1:if(!b)return E.return(c);h={};e=0;f=I(Object.keys(b));for(k=f.next();!k.done;k=f.next())m=k.value,n=b[m],"string"!==typeof n&&"texture"===n.type&&++e;1<e&&(t.v=!1);r=I(Object.keys(b));k=r.next();case 2:if(k.done){E.g=4;break}p=k.value;u=b[p];if("string"===typeof u)return M=h,pa=p,T(E,Ma(t,p,c[u]),15);l=c[u.stream];if(void 0===l){E.g=3;break}if("detection_list"===u.type){var x=
l.getRectList(),J=l.getLandmarksList(),F=l.getClassificationsList(),H=[];if(x)for(var N=0;N<x.size();++N){var da={R:x.get(N),M:Ca(J.get(N)),L:Da(F.get(N))};H.push(da)}h[p]=H;E.g=7;break}if("landmarks"===u.type){q=l.getLandmarks();h[p]=q?Ca(q):void 0;E.g=7;break}if("landmarks_list"===u.type){if(v=l.getLandmarksList())for(x=[],J=v.size(),F=0;F<J;++F)H=v.get(F),x.push(Ca(H)),H.delete();else x=void 0;h[p]=x;E.g=7;break}if("rect_list"===u.type){if(w=l.getRectList())for(x=[],J=w.size(),F=0;F<J;++F)H=w.get(F),
x.push(H);else x=void 0;h[p]=x;E.g=7;break}if("classifications_list"===u.type){if(y=l.getClassificationsList())for(x=[],J=y.size(),F=0;F<J;++F)H=y.get(F),x.push(Da(H));else x=void 0;h[p]=x;E.g=7;break}if("object_detection_list"===u.type){if(z=l.getObjectDetectionList())for(x=[],J=z.size(),F=0;F<J;++F){H=z.get(F);N=x;da=N.push;for(var Pa=H.id,qa=H.keypoints,ra=[],Qa=qa.size(),ea=0;ea<Qa;++ea){var P=qa.get(ea);ra.push({id:P.id,point3d:{x:P.point3d.x,y:P.point3d.y,z:P.point3d.z},point2d:{x:P.point2d.x,
y:P.point2d.y,depth:P.point2d.depth}})}da.call(N,{id:Pa,keypoints:ra,visibility:H.visibility})}else x=void 0;h[p]=x;E.g=7;break}if("texture"!==u.type)throw Error("Unknown output config type: '"+u.type+"'");A=t.s[p];A||(A=new ua(t.h,t.A,t.v),t.s[p]=A);x=A;J=l.getTexture2d();wa(x,J);x=va(x);return T(E,x,14);case 14:O=E.h,h[p]=O;case 7:u.transform&&h[p]&&(h[p]=u.transform(h[p]));E.g=3;break;case 15:M[pa]=E.h;case 3:k=r.next();E.g=2;break;case 4:return E.return(h)}})})}
function Ma(a,c,b){return Y(a,function g(){var h=this,e;return W(g,function(f){if(b.isNumber())return f.return(b.getNumber());if(b.isRect())return f.return(b.getRect());if(b.isLandmarks())return f.return(b.getLandmarks());if(b.isLandmarksList())return f.return(b.getLandmarksList());if(b.isClassificationsList())return f.return(b.getClassificationsList());if(b.isObjectDetectionList())return f.return(b.getObjectDetectionList());if(b.isTexture2d()){e=h.s[c];e||(e=new ua(h.h,h.A,h.v),h.s[c]=e);var k=f.return;
var m=e;var n=b.getTexture2d();wa(m,n);m=va(m);return k.call(f,m)}return f.return(void 0)})})}
function Ja(a,c){for(var b=c.name||"$",d=[].concat(fa(c.wants)),g=new a.h.StringList,h=I(c.wants),e=h.next();!e.done;e=h.next())g.push_back(e.value);h=a.h.PacketListener.implement({onResults:function(f){return Y(a,function m(){var n=this,t,r,p,u,l;return W(m,function(q){t=n;r={};for(p=0;p<c.wants.length;++p)r[d[p]]=f.get(p);u=La(n,r,c.outs);if(l=n.listeners[b])return q.return(n.F.then(function(){return u}).then(function(v){return Y(t,function y(){var z,A,O=this;return W(y,function(M){z=l(v);for(A=
0;A<c.wants.length;++A)r[d[A]].delete();if(z)return O.F=z,M.return(z);M.g=0})})}));q.g=0})})}});a.i.attachMultiListener(g,h);g.delete()}B.onResults=function(a,c){this.listeners[c||"$"]=a};X("Solution",Fa);X("OptionType",{NUMBER:0,BOOL:1,0:"NUMBER",1:"BOOL"});function Na(a){void 0===a&&(a=0);switch(a){case 1:return"pose_landmark_full.tflite";case 2:return"pose_landmark_heavy.tflite";default:return"pose_landmark_lite.tflite"}}
function Oa(a){var c=this;a=a||{};this.g=new Fa({locateFile:a.locateFile,files:function(b){return[{url:"pose_solution_packed_assets_loader.js"},{simd:!1,url:"pose_solution_wasm_bin.js"},{simd:!0,url:"pose_solution_simd_wasm_bin.js"},{data:!0,url:Na(b.modelComplexity)}]},graph:{url:"pose_web.binarypb"},listeners:[{wants:["pose_landmarks","world_landmarks","image_transformed"],outs:{image:"image_transformed",poseLandmarks:{type:"landmarks",stream:"pose_landmarks"},poseWorldLandmarks:{type:"landmarks",
stream:"world_landmarks"}}}],inputs:{image:{type:"video",stream:"input_frames_gpu"}},options:{selfieMode:{type:1,graphOptionXref:{calculatorType:"GlScalerCalculator",calculatorIndex:1,fieldName:"flip_horizontal"}},modelComplexity:{type:0,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorModelComplexity",fieldName:"int_value"},onChange:function(b){return Y(c,function g(){var h,e,f=this,k;return W(g,function(m){if(1==m.g)return h=Na(b),e="third_party/mediapipe/modules/pose_landmark/"+
h,T(m,Z(f.g,h),2);k=m.h;f.g.overrideFile(e,k);return m.return(!0)})})}},smoothLandmarks:{type:1,graphOptionXref:{calculatorType:"ConstantSidePacketCalculator",calculatorName:"ConstantSidePacketCalculatorSmoothLandmarks",fieldName:"bool_value"}},minDetectionConfidence:{type:0,graphOptionXref:{calculatorType:"TensorsToDetectionsCalculator",calculatorName:"poselandmarkgpu__posedetectiongpu__TensorsToDetectionsCalculator",fieldName:"min_score_thresh"}},minTrackingConfidence:{type:0,graphOptionXref:{calculatorType:"ThresholdingCalculator",
calculatorName:"poselandmarkgpu__poselandmarkbyroigpu__poselandmarkbyroipostprocessing__ThresholdingCalculator",fieldName:"threshold"}}}})}B=Oa.prototype;B.reset=function(){this.g.reset()};B.close=function(){this.g.close();return Promise.resolve()};B.onResults=function(a){this.g.onResults(a)};B.initialize=function(){return Y(this,function c(){var b=this;return W(c,function(d){return T(d,b.g.initialize(),0)})})};
B.send=function(a,c){return Y(this,function d(){var g=this;return W(d,function(h){return T(h,g.g.send(a,c),0)})})};B.setOptions=function(a){this.g.setOptions(a)};X("Pose",Oa);X("POSE_CONNECTIONS",[[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]]);
X("POSE_LANDMARKS",{NOSE:0,LEFT_EYE_INNER:1,LEFT_EYE:2,LEFT_EYE_OUTER:3,RIGHT_EYE_INNER:4,RIGHT_EYE:5,RIGHT_EYE_OUTER:6,LEFT_EAR:7,RIGHT_EAR:8,LEFT_RIGHT:9,RIGHT_LEFT:10,LEFT_SHOULDER:11,RIGHT_SHOULDER:12,LEFT_ELBOW:13,RIGHT_ELBOW:14,LEFT_WRIST:15,RIGHT_WRIST:16,LEFT_PINKY:17,RIGHT_PINKY:18,LEFT_INDEX:19,RIGHT_INDEX:20,LEFT_THUMB:21,RIGHT_THUMB:22,LEFT_HIP:23,RIGHT_HIP:24,LEFT_KNEE:25,RIGHT_KNEE:26,LEFT_ANKLE:27,RIGHT_ANKLE:28,LEFT_HEEL:29,RIGHT_HEEL:30,LEFT_FOOT_INDEX:31,RIGHT_FOOT_INDEX:32});
X("POSE_LANDMARKS_LEFT",{LEFT_EYE_INNER:1,LEFT_EYE:2,LEFT_EYE_OUTER:3,LEFT_EAR:7,LEFT_RIGHT:9,LEFT_SHOULDER:11,LEFT_ELBOW:13,LEFT_WRIST:15,LEFT_PINKY:17,LEFT_INDEX:19,LEFT_THUMB:21,LEFT_HIP:23,LEFT_KNEE:25,LEFT_ANKLE:27,LEFT_HEEL:29,LEFT_FOOT_INDEX:31});
X("POSE_LANDMARKS_RIGHT",{RIGHT_EYE_INNER:4,RIGHT_EYE:5,RIGHT_EYE_OUTER:6,RIGHT_EAR:8,RIGHT_LEFT:10,RIGHT_SHOULDER:12,RIGHT_ELBOW:14,RIGHT_WRIST:16,RIGHT_PINKY:18,RIGHT_INDEX:20,RIGHT_THUMB:22,RIGHT_HIP:24,RIGHT_KNEE:26,RIGHT_ANKLE:28,RIGHT_HEEL:30,RIGHT_FOOT_INDEX:32});X("POSE_LANDMARKS_NEUTRAL",{NOSE:0});}).call(this);
