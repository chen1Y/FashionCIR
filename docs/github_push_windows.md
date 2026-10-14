# 在这台 Windows 电脑推送实验分支

当前仓库的 `origin` 是 `https://github.com/chen1Y/FashionCIR.git`，实验分支为
`codex/multi-positive-cir`。本机已安装 Git Credential Manager（GCM 2.7.0），
Git 的系统配置也已有 `credential.helper=manager`，但尚未找到可用的 GitHub 登录凭据。

另一个已验证可用的路径是实验服务器：其 Git remote 使用 GitHub SSH，推送预检查成功。
本次可以通过服务器现有认证发布实验分支，无需复制服务器的私钥到本机。
下面的浏览器授权步骤用于以后直接从这台 Windows 电脑推送。

在 PowerShell 中运行：

```powershell
Set-Location 'C:\Users\Lee\Documents\ChatGPT\FashionCIR'
& 'C:\Program Files\Git\mingw64\bin\git-credential-manager.exe' github login --username chen1Y --browser
git push -u origin codex/multi-positive-cir
```

第一条登录命令会使用浏览器：由本人登录拥有仓库写权限的 GitHub 账号并确认授权，
随后 GCM 将凭据保存在 Windows 的凭据存储中。完成登录后再执行 push。
服务器的 SSH 登录密码不能用于 GitHub 身份认证。

成功后可以在仓库页面的分支选择器切换到 `codex/multi-positive-cir`。
推送分支不会自动修改 `main`；需要正式采用时，再创建 Pull Request 对比并合并。

检查状态（不打印 token）：

```powershell
& 'C:\Program Files\Git\mingw64\bin\git-credential-manager.exe' github list
git status -sb
git log -1 --oneline
git ls-remote --heads origin codex/multi-positive-cir
```

如果浏览器登录失败，可尝试官方设备授权流程：

```powershell
& 'C:\Program Files\Git\mingw64\bin\git-credential-manager.exe' github login --username chen1Y --device --no-ui
```

按命令实际显示的地址和一次性代码，在自己的浏览器中完成授权。
本次自动尝试的设备流程未返回授权码，已停止该进程；不能据此认为已登录。
只有当认证成功且账号对该仓库有写权限时，push 才能成功。

GitHub 官方说明：
[Git 凭据缓存与 GCM](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git)、
[GitHub 身份认证方式](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)。

## 本地提交经服务器推送

若服务器已拥有本地分支的起点 `main`，可用 bundle 传递新增提交：

```powershell
git bundle create .autodl/multi-positive.bundle main..codex/multi-positive-cir
scp -P 18175 .autodl/multi-positive.bundle root@connect.weste.seetacloud.com:/root/autodl-tmp/multi-positive.bundle
```

在服务器执行：

```bash
git -C /root/autodl-tmp fetch /root/autodl-tmp/multi-positive.bundle \
  refs/heads/codex/multi-positive-cir:refs/heads/codex/multi-positive-cir
git -C /root/autodl-tmp push origin codex/multi-positive-cir
```

fetch 创建实验分支引用，push 发布该分支；无需 checkout，也不会把服务器 `main` 的工作目录切换过去。
SSH 连接使用服务器登录认证；`git push` 使用服务器已有的 GitHub SSH 认证，两者是独立凭据。
