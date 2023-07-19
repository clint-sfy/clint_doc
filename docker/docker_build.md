## Docker 应用部署

### 一、部署MySQL

1. 搜索mysql镜像

```shell
docker search mysql
```

2. 拉取mysql镜像

```shell
docker pull mysql:5.6
```

3. 创建容器，设置端口映射、目录映射

```shell
# 在/root目录下创建mysql目录用于存储mysql数据信息
mkdir ~/mysql
cd ~/mysql

docker network create common-network
//查看网络
docker network ls
//查看网络容器
docker network inspect common-network
```

```shell
docker run -id \
-p 3306:3306 \
--name=c_mysql \
-v $PWD/conf:/etc/mysql/conf.d \
-v $PWD/logs:/logs \
-v $PWD/data:/var/lib/mysql \
-e MYSQL_ROOT_PASSWORD=123456 \
mysql:5.7

docker run -p 3306:3306 --name mysql -d \
--restart=always \
--network common-network \
-v $PWD/conf:/etc/mysql/conf.d \
-v $PWD/logs:/logs \
-v $PWD/data:/var/lib/mysql \
-e MYSQL_ROOT_PASSWORD=admin \
mysql:5.7 

docker run -id \
-p 3306:3306 \
--name=mysql8 \
-v $PWD/conf:/etc/mysql/conf.d \
-v $PWD/logs:/logs \
-v $PWD/data:/var/lib/mysql \
-e MYSQL_ROOT_PASSWORD=123456 \
mysql:8.0.28


```

- 参数说明：
  - **-p 3307:3306**：将容器的 3306 端口映射到宿主机的 3307 端口。
  - **-v $PWD/conf:/etc/mysql/conf.d**：将主机当前目录下的 conf/my.cnf 挂载到容器的 /etc/mysql/my.cnf。配置目录
  - **-v $PWD/logs:/logs**：将主机当前目录下的 logs 目录挂载到容器的 /logs。日志目录
  - **-v $PWD/data:/var/lib/mysql** ：将主机当前目录下的data目录挂载到容器的 /var/lib/mysql 。数据目录
  - **-e MYSQL_ROOT_PASSWORD=123456：**初始化 root 用户的密码。



4. 进入容器，操作mysql

```shell
docker exec –it c_mysql /bin/bash
```

5. 使用外部机器连接容器中的mysql

![1573636765632](.\imgs\1573636765632.png)


















### 二、部署Tomcat

1. 搜索tomcat镜像

```shell
docker search tomcat
```

2. 拉取tomcat镜像

```shell
docker pull tomcat
```

3. 创建容器，设置端口映射、目录映射

```shell
# 在/root目录下创建tomcat目录用于存储tomcat数据信息
mkdir ~/tomcat
cd ~/tomcat
```

```shell
docker run -id --name=c_tomcat \
-p 8080:8080 \
-v $PWD:/usr/local/tomcat/webapps \
tomcat 
```

- 参数说明：
  - **-p 8080:8080：**将容器的8080端口映射到主机的8080端口
  
    **-v $PWD:/usr/local/tomcat/webapps：**将主机中当前目录挂载到容器的webapps



4. 使用外部机器访问tomcat

![1573649804623](./imgs\1573649804623.png)
















### 三、部署Nginx

1. 搜索nginx镜像

```shell
docker search nginx
```

2. 拉取nginx镜像

```shell
docker pull nginx
```

3. 创建容器，设置端口映射、目录映射


```shell
# 在/root目录下创建nginx目录用于存储nginx数据信息
mkdir ~/nginx
cd ~/nginx
mkdir conf
cd conf
# 在~/nginx/conf/下创建nginx.conf文件,粘贴下面内容
vim nginx.conf
```
```shell

user  nginx;
worker_processes  1;

error_log  /var/log/nginx/error.log warn;
pid        /var/run/nginx.pid;


events {
    worker_connections  1024;
}


http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile        on;
    #tcp_nopush     on;

    keepalive_timeout  65;

    #gzip  on;

    include /etc/nginx/conf.d/*.conf;
}


```




```shell
docker run -id --name=c_nginx \
-p 80:80 \
-v $PWD/conf/nginx.conf:/etc/nginx/nginx.conf \
-v $PWD/logs:/var/log/nginx \
-v $PWD/html:/usr/share/nginx/html \
nginx
```

- 参数说明：
  - **-p 80:80**：将容器的 80端口映射到宿主机的 80 端口。
  - **-v $PWD/conf/nginx.conf:/etc/nginx/nginx.conf**：将主机当前目录下的 /conf/nginx.conf 挂载到容器的 :/etc/nginx/nginx.conf。配置目录
  - **-v $PWD/logs:/var/log/nginx**：将主机当前目录下的 logs 目录挂载到容器的/var/log/nginx。日志目录

4. 使用外部机器访问nginx

![1573652396669](.\imgs\1573652396669.png)











### 四、部署Redis

1. 搜索redis镜像

```shell
docker search redis
```

2. 拉取redis镜像

```shell
docker pull redis:5.0
```

3. 创建容器，设置端口映射

```shell
docker run -id --name=c_redis -p 6379:6379 redis:5.0
```

4. 使用外部机器连接redis

```shell
./redis-cli.exe -h 192.168.149.135 -p 6379
```

### 五、部署nacos

```
docker network create common-network
//查看网络
docker network ls
//查看网络容器
docker network inspect common-network

docker pull nacos/nacos-server

docker run --network common-network --env MODE=standalone --name nacos -d -p 8848:8848 nacos/nacos-server
// 进入容器
docker exec -it nacos bash
// 修改容器配置
cd conf
vi application.properties

http://ip:8848/nacos/index.html
nacos/nacos(用户名密码)
```

```
docker run -d \
-e MODE=standalone \
-e SPRING_DATASOURCE_PLATFORM=mysql \
-e MYSQL_SERVICE_HOST=172.18.0.3 \
-e MYSQL_SERVICE_PORT=3306 \
-e MYSQL_SERVICE_USER=root \
-e MYSQL_SERVICE_PASSWORD=123456 \
-e MYSQL_SERVICE_DB_NAME=xdclass_nacos \
-p 8848:8848 \
--restart=always \
--name nacos \
nacos/nacos-server 
```

```
INSERT INTO `users` (`username`, `password`, `enabled`)
VALUES
	('nacos', '$2a$10$EuWPZHzz32dJN7jexM34MOeYirDdFAZm2kuWj7VEOJhhZkDrxfvUu', 1);
```

### 六、部署RabbitMQ

- 阿里云安装RabbitMQ
  - 阿里云折扣地址 https://www.aliyun.com/minisite/goods?userCode=r5saexap&share_source=copy_link
  - 最少 2核4g或者推荐 2核8g（用家人账号购买，接近1折，初次买1年或者3年）
- 登录个人的Linux服务器
  - ssh root@8.129.113.233
- Docker安装RabbitMQ
  - 地址：https://hub.docker.com/_/rabbitmq/

```
#拉取镜像
docker pull rabbitmq:3.8.12-management-alpine

docker run -d --hostname rabbit_host1 --name xd_rabbit -e RABBITMQ_DEFAULT_USER=admin -e RABBITMQ_DEFAULT_PASS=password -p 15672:15672 -p 5672:5672 rabbitmq:3.8.12-management-alpine

#介绍
-d 以守护进程方式在后台运行
-p 15672:15672 management 界面管理访问端口
-p 5672:5672 amqp 访问端口
--name：指定容器名
--hostname：设定容器的主机名，它会被写到容器内的 /etc/hostname 和 /etc/hosts，作为容器主机IP的别名，并且将显示在容器的bash中

-e 参数
  RABBITMQ_DEFAULT_USER 用户名
  RABBITMQ_DEFAULT_PASS 密码
```

* 主要端口介绍

```
4369 erlang 发现口

5672 client 端通信口

15672 管理界面 ui 端口

25672 server 间内部通信口
```

- 访问管理界面

  - ip:15672

- 注意事项！！！！

  - Linux服务器检查防火墙是否关闭
  - 云服务器检查网络安全组是否开放端口

  ```
  CentOS 7 以上默认使用的是firewall作为防火墙
  查看防火墙状态
  firewall-cmd --state
  
  停止firewall
  systemctl stop firewalld.service
  
  禁止firewall开机启动
  systemctl disable firewalld.service
  ```

  









