// 这里需要注意，文件命名必需为index.js，且放在根目录下
// 因为尝试过其它的命名都没有生效

import express from "express";
import process from "node:process";

const app = express();

app.get('/', (request, response) => {
    response.send('Hello World!');
});

// 监听端口
const port = process.env.PORT || 4000;
app.listen(port, () => {
    const header = "Server is now running at:";
    const local = `- Local: http://localhost:${port}`;
    console.log([header, local].join("\n"));
});

export default app;