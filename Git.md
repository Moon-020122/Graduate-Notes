 [GitCheatSheet_byGeekHour_v1.0.0.pdf](GitCheatSheet_byGeekHour_v1.0.0.pdf) 

# 设置个人信息

`git config --global user.name "姓名"`  如果姓名中无空格可省略双引号

`git config --global user.email 邮箱地址`

`git config --global --list`

![image-20240522144132936](C:\Users\12625\AppData\Roaming\Typora\typora-user-images\image-20240522144132936.png)

# 新建版本库

方式一 `git init` 后方可添加名称，则会在当前目录下再创建一个目录来作为git仓库`git init my-repo`

方式二 `git clone + 项目地址`

# 本地区域和文件状态

工作区(Working Directory)：本地目录，在资源管理器中看见的文件夹就是工作区。

暂存区(Staging Area/Index)：临时存储区域，用于保存即将提交到git仓库的修改内容。

​	使用`git ls-files`查看暂存区中的文件

本地仓库(Local Repository)：上节课使用git init 命令所创建的仓库，git存储代码和版本信息的主要位置。

![image-20240522145600350](C:\Users\12625\AppData\Roaming\Typora\typora-user-images\image-20240522145600350.png)

未跟踪(Untrack)：新创建未被git管理的文件。

未修改(Unmodified)：被git管理的文件，但文件内容无变化。

已修改(Modified)：已经修改的文件暂未添加进暂存区。

已暂存(Staged)：修改后且添加到暂存区内的文件。

![image-20240522145939399](C:\Users\12625\AppData\Roaming\Typora\typora-user-images\image-20240522145939399.png)

# 添加和提交文件

`git status`查看当前仓库的状态，如查看当前仓库处在哪个分支、有哪些文件以及当前文件处在什么状态。

`git add 文件名`将文件添加到暂存区等待后续提交操作。

​	通配符`git add *.txt（例子）` 可以将所有以.txt文件添加到暂存区内。

​	`git add .` 将当前文件夹下的所有文件都添加进暂存区里面

` git commit -m "提交备注"`只会提交暂存区的文件而不会提交工作区的其他文件 ，当不使用-m来编辑提交信息时，默认会使用vim来编辑提交信息(i进入编辑模式，esc回到命令模式，输入:wq保存退出)

`git log`来查看提交信息。前方会有提交编号，是对应的版本ID，用以对版本进行管理。

​	`git log --oneline` 查看简洁信息

# git reset回退版本

`git reset 版本ID` 回退版本，HEAD^`可表示上一个版本。

​	`git reset --soft` 回退到某一个版本并保留工作区和暂存区的所有修改内容。

​	`git reset --hard`回退到某一个版本并丢弃工作区和暂存区的所有修改内容。

​	`git reset --mixed`回退到某一个版本保留工作区内容而丢弃暂存区的修改内容(默认参数)。

![image-20240522151642836](C:\Users\12625\AppData\Roaming\Typora\typora-user-images\image-20240522151642836.png)

# git reflog命令回溯

`git reflog`可以查看所执行的操作，用以误操作后找不到版本ID等情况。

# git diff 查看差异

`git diff`可以用来查看文件在工作区、暂存区以及版本库之间的差异；查看两个文件不同版本的差异；查看两个文件不同分支之间的差异。

​	`git diff` 默认比较工作区与暂存区之间的内容，显示发生更改的文件以及更改的详细信息。 

​		第一行为差异文件；第二行为四十位哈希值以及文件权限；红色为删除内容，绿色为添加内容。

![image-20240522153001503](C:\Users\12625\AppData\Roaming\Typora\typora-user-images\image-20240522153001503.png)

​	`git diff HEAD` 比较工作区与版本库之间的差异。

​	`git diff --cached` 比较暂存区与版本库之间的差异。

​	`git diff VID1 VID2` 比较版本1和版本2之间的差异。

​	`git diff VID1 HEAD` 比较版本1和最新版本之间的差异。

​	`git diff HEAD~/HEAD^/HEAD~2(上上个 版本) HEAD` 比较上一个版本和最新版本之间的差异。

​	`git diff XXX XXX 文件名.后缀` 比较当前文件之间的差异。

# git rm删除文件

## 第一种删除方式

 直接删除工作区的文件再添加到暂存区然后提交。

## 第二在删除方式

`git rm 文件名`此操作会直接从工作区和暂存区中同时删除，仅需要提交即可。

`gir rm --cached <file>` 把文件从暂存区中删除，但保留在当前工作区中。

`gir rm -r *` 递归删除某个目录下的所有子目录和文件。

# .gitignore忽略文件(特殊文件)

​	这个文件的作用是可以让我们忽略掉一些不应该被加入到版本库中的文件。 

![image-20240522154739915](C:\Users\12625\AppData\Roaming\Typora\typora-user-images\image-20240522154739915.png)

 	在.gitignore中直接添加文件名。 

​	*.log忽略所有.log后缀的文件。

​	temp/胡罗temp文件夹中的文件，文件夹是以/结尾。

## 文件匹配规则

![image-20240522155556225](C:\Users\12625\AppData\Roaming\Typora\typora-user-images\image-20240522155556225.png)

第三个和第五个都表示忽略当前目录下的文件或文件夹而不忽略子目录下的相应文件或文件夹。

![image-20240522155722373](C:\Users\12625\AppData\Roaming\Typora\typora-user-images\image-20240522155722373.png)

官方忽略模板

[github/gitignore: A collection of useful .gitignore templates](https://github.com/github/gitignore/)

# 将本地仓库与远程仓库关联

ssh连接具体看视频。

## git push将本地仓库上传到远程仓库

`git remote add origin git@github.com:Moon-020122/test.git`原型为`git remote add <shortname> <url>`
`git branch -M main`
`git push -u origin main`要注意push的时候两分支名要相同。不相同时候要注意使用`git push -u origin main:分支名称`

## git pull 将远程仓库下载到本地仓库

 `git pull <远程仓库名> <远程仓库分支>:<本地分支名>` 其中仓库名和分支名可省略，默认拉取仓库别名origin的main分支，在拉取远程仓库内容到本地仓库时会自动合并，如何没有冲突则会合并成功，如果有冲突则会失败，需要我们手动解决掉冲突。

`git fetch` 只获取远程仓库的修改，但并不自动合并到本地仓库中，需要手动合并。

# 分支

`git branch`可以查看当前的所有分支

​	`git branch <分支名>` 创建一个新的分支，并没有切换到这个分支上。

​	`git checkout <分支名>` 切换到对应分支 但`checkout`命令也可用于回溯操作，当文件名与分支名冲突时会认	为此操作为切换分支，因此有一个专门用于切换分支的命令`git switch <分支名>`

​	`git branch -d <分支名>`删除对应已经合并的分支，如果未合并而要求删除，则使用`-D`强制删除。

`git merge <分支名>` 合并分支，merge后面的分支名称是将要被合并的分支，当前所在分支就算合并后的目标分支。

## 分支冲突解决

如出现两个分支同时修改同一行代码，合并时就会出错。使用`git diff`或者`git status`可查看冲突内容。

# rebase和回退

`rebase`可以在任意分支上执行操作，区别是顺序有所不同。

![image-20240522185231663](C:\Users\12625\AppData\Roaming\Typora\typora-user-images\image-20240522185231663.png)

`merge`优缺点：

​	优点：不会破环原分支的提交历史，方便回溯和查看。

​	缺点：会产生额外的提交节点，分支图复杂。

`rebase`优缺点：

​	优点：不会新增额外的提交记录，形成线性历史，直观干净。

​	缺点：会改变提交历史，改变了当前分支branch out的节点，应避免在共享分支使用。

# 主要流程

`git init`

`git add . `

`git commit -m <注释>`

`git branch -M main`

`git remote add origin <SSH Address>`

`git push -u origin main`

