### 一、安装Docker



```shell
# 1、yum 包更新到最新 
yum update
# 2、安装需要的软件包， yum-util 提供yum-config-manager功能，另外两个是devicemapper驱动依赖的 
yum install -y yum-utils device-mapper-persistent-data lvm2
# 3、 设置yum源
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
# 4、 安装docker，出现输入的界面都按 y 
yum install -y docker-ce
# 5、 查看docker版本，验证是否验证成功
docker -v

```
###  常用命令
```
systemctl start docker
systemctl enable docker
```

```
docker images
docker images -q
```

```
docker search redis
docker pull XXX	
docker rmi ID
docker rmi `docker images -q`
```

```
docker run -it --name=cl centos:7 /bin/bash 
-i 一直运行
-t 分配一个终端
-d 后台运行 不会自动关闭

docker run -id --name=cl centos:7
docker exec -it c1 /bin/bash
```

```
docker ps
docker ps -a 历史容器
```

```
docker start c1
docker stop c1
docker rm `docker ps -aq`
```

```
docker inspect c1  查看容器信息
```

### 数据卷

是宿主机中的一个目录或文件

当绑定后，修改会同步

```

docker run   -v  宿主机文件:容器内目录

目录必须是绝对路径
```

```
配置数据卷
--name=c1 -v /volume
--volumes-from c1
--volumes-from c1
```

### 应用部署















