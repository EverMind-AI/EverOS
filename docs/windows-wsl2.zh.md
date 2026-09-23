# 在 Windows 上跑 EverOS（WSL2）

EverOS 的开发和 CI 都在 Linux 和 macOS 上。Windows 上走 WSL2。

WSL2 是 Windows 自带的一套真 Linux 内核，不是模拟器也不是虚拟机软件。装完之后
EverOS 就跟跑在一台 Ubuntu 服务器上没区别，文件锁、向量索引、文件监听全部是
Linux 原生行为。所以这条路的好处不是「能跑」，是**它和我们测过的环境完全一致**。

这篇写完整的安装过程，包括脚本绕不过去的那一步，和一个不看文档必踩、踩了还没有
任何报错的坑。

## 目录

- [开始之前](#开始之前)
  - [这台机器要不要重启](#这台机器要不要重启)
- [安装](#安装)
  - [第一步：装 WSL2 和 Ubuntu](#第一步装-wsl2-和-ubuntu)
  - [第二步：在 Ubuntu 里装 EverOS](#第二步在-ubuntu-里装-everos)
  - [一个脚本跑完](#一个脚本跑完)
- [确认装好了](#确认装好了)
- [数据目录千万别放 C 盘](#数据目录千万别放-c-盘)
- [Office 文档支持](#office-文档支持)
- [从 Windows 这边访问 API](#从-windows-这边访问-api)
- [出问题了](#出问题了)
- [直接在 Windows 上装行不行](#直接在-windows-上装行不行)

## 开始之前

| 要求 | 怎么确认 |
|---|---|
| Windows 11，或 Windows 10 build 19041 以上 | 运行 `winver` |
| 管理员权限 | 开 WSL 功能需要提权 |
| CPU 虚拟化已在固件里打开 | 任务管理器 → 性能 → CPU → 「虚拟化：已启用」 |
| 磁盘 3 GB 起，装 LibreOffice 再加 1 GB | |

公司发的电脑上，Hyper-V 和虚拟机平台有可能被组策略锁死。这种情况没有绕法，得找 IT。

### 这台机器要不要重启

`wsl --install` 要打开 Windows 的 `VirtualMachinePlatform` 功能，而打开它需要重启。

但这个功能**很多机器上本来就是开的** —— 装过 Docker Desktop、开过 Hyper-V、用过
Windows 沙盒或安卓模拟器、公司推过基于虚拟化的安全策略，任意一条命中就已经开了。
先查一下，别白白安排一次重启：

```powershell
(Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State
```

- 结果是 `Enabled` → 不用重启，整个安装一口气跑完。
- 结果是 `Disabled` → 要重启一次，而且只有这一次，以后升级 EverOS 都不用。

## 安装

### 第一步：装 WSL2 和 Ubuntu

用**管理员身份**打开 PowerShell：

```powershell
wsl --install --no-launch
```

`--no-launch` 不能省。不加的话 `wsl --install` 装完会直接把 Ubuntu 拉起来，停在
「Enter new UNIX username」等你输用户名 —— 脚本就卡死在这儿了。

如果上面预检查出来是 `Disabled`，现在重启。然后创建发行版，同样跳过交互：

```powershell
wsl --install -d Ubuntu --no-launch
ubuntu install --root
```

### 第二步：在 Ubuntu 里装 EverOS

```bash
wsl -d Ubuntu -u root -- bash -lc '
  set -euo pipefail
  apt-get update -qq
  apt-get install -y -qq python3-venv curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 /opt/everos
  uv pip install --python /opt/everos/bin/python everos
'
```

### 一个脚本跑完

上面两步可以合成一个脚本。它是幂等的 —— 重启之后原样再跑一遍，会从断掉的地方接上，
不会重复装。

```powershell
# everos-setup.ps1 —— 用管理员身份运行
$ErrorActionPreference = 'Stop'

# 第一步：WSL 平台
if (-not (Get-Command wsl -ErrorAction SilentlyContinue) -or -not (wsl --version 2>$null)) {
    wsl --install --no-launch
    Write-Host 'WSL 装好了。重启电脑，然后再跑一遍这个脚本。' -ForegroundColor Yellow
    exit 0
}

# 第二步：Ubuntu
if (-not (wsl -l -q | Select-String -Quiet 'Ubuntu')) {
    wsl --install -d Ubuntu --no-launch
    ubuntu install --root
}

# 第三步：EverOS
wsl -d Ubuntu -u root -- bash -lc @'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq python3-venv curl
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 /opt/everos
uv pip install --python /opt/everos/bin/python everos
'@

Write-Host '装完了。启动：wsl -d Ubuntu -- /opt/everos/bin/everos serve' -ForegroundColor Green
```

判断 WSL 装没装用 `wsl --version`，别用 `wsl --status` —— 后者的退出码在几种情况下
都是 0，判断不出来。

## 确认装好了

```bash
wsl -d Ubuntu -- /opt/everos/bin/everos --version
wsl -d Ubuntu -- /opt/everos/bin/everos init
wsl -d Ubuntu -- /opt/everos/bin/everos serve
```

到这儿就可以照着 [快速开始](../README.md#quick-start) 往下走了。进了 Ubuntu 之后，
所有操作和在一台普通 Ubuntu 机器上完全一样。

## 数据目录千万别放 C 盘

> [!IMPORTANT]
> EverOS 存记忆的目录（memory root）要放在 Ubuntu 自己的文件系统里，比如
> `/home/...` 或 `/opt/...`。**不要放在 `/mnt/c/` 底下。**

这是这条路上唯一一个**不报错的坑**，所以单独拿出来讲。

EverOS 有个后台组件一直盯着这个目录里的 md 文件，文件一改就更新索引。而 Windows
那边的文件改动**不会通知到 WSL2 里面** —— 这是 WSL2 的机制限制，不是 EverOS 的 bug。

后果是：目录放在 `/mnt/c` 上时，这个监听组件会正常启动、正常打日志、看起来一切健康，
但一个文件事件都收不到。你改了 md 文件，索引不动；然后搜索安安静静地返回旧数据。
日志里没有任何一行提示这件事。

如果因为别的原因必须把文件放在 Windows 那边，就只能退回定时扫描：

| 办法 | 怎么做 |
|---|---|
| 用默认的定时扫描（30 秒一轮） | 本来就开着，慢一点但最终会同步上 |
| 把间隔调短 | 目录不大的话调到 5 秒左右 |
| 改完手动同步一次 | `everos cascade sync` |

细节见 [cascade 运维手册](cascade_runbook.md#wsl2--network-mounts)。

## Office 文档支持

要解析 Word / Excel / PPT，得在 Ubuntu 里装 **Linux 版** LibreOffice：

```bash
wsl -d Ubuntu -u root -- apt-get install -y libreoffice
```

Windows 版的 LibreOffice 装了也没用。EverOS 是在 Ubuntu 内部用 `shutil.which("soffice")`
去 PATH 里找这个程序的，找不到 Windows 那边的 `soffice.exe`。

没装的话，上传 Office 文件会返回 `503 CAPABILITY_UNAVAILABLE`。其他格式（图片、
PDF、音频）不受影响，见 [multimodal.md](multimodal.md#libreoffice-office-documents-only)。

## 从 Windows 这边访问 API

能访问，不用额外配置。WSL2 默认开着 localhost 转发，Ubuntu 里监听的端口，Windows
上用同一个端口就能连。`everos serve` 在 Ubuntu 内部绑的是 `127.0.0.1`，转发照样能
中继过去，所以这个默认值不用改：

```bash
wsl -d Ubuntu -- /opt/everos/bin/everos serve   # 在 Ubuntu 里监听
curl http://localhost:8000/health               # Windows 这边直接连
```

连不上的话，最常见的两个原因是 VPN 客户端接管了 WSL 的网络，或者转发进程卡住了。
先重启一下 WSL：

```powershell
wsl --shutdown        # 下次跑 wsl 命令会自动重启，转发也跟着重建
```

> [!WARNING]
> 别拿 `host = "0.0.0.0"` 当解决办法。**EverOS 自己不带任何鉴权** —— 绑回环地址
> 就是它保持私有的唯一手段（见 [SECURITY.md](../SECURITY.md)）。而且 `0.0.0.0`
> 到底暴露到哪，取决于 WSL 的网络模式，两种差别很大：
>
> | `.wslconfig` 里的 `networkingMode` | 绑 `0.0.0.0` 会暴露给谁 |
> |---|---|
> | `nat`（默认） | 只到 Ubuntu 的虚拟网段和 Windows 本机。局域网里的其他机器连不上，除非你手动配了 `netsh portproxy`。 |
> | `mirrored`（WSL 2.0 以上） | Windows 这台机器的**所有网卡** —— 这个无鉴权的 API 就挂到局域网上了。 |
>
> 只有在自己的网关或鉴权层已经挡在前面时，才用 `0.0.0.0`。

## 出问题了

| 你看到的现象 | 原因 | 怎么办 |
|---|---|---|
| `wsl --install` 卡在 「Enter new UNIX username」 | 漏了 `--no-launch` | Ctrl-C，加上 `--no-launch` 重跑，然后 `ubuntu install --root` |
| 提示「无法启动虚拟机」 | 固件里虚拟化没开 | 进 BIOS/UEFI 打开 VT-x 或 AMD-V |
| 报错 `WslRegisterDistribution failed with error: 0x80370102` | 同上，或者 Hyper-V 被组策略禁了 | 改固件设置，或找 IT |
| 重启完就没动静了 | 脚本没再跑一遍 | 再跑一遍，它是幂等的 |
| 改了 md 文件，搜索还是旧内容 | 数据目录放在 `/mnt/c` 上了 | 挪进 Ubuntu 的文件系统 |
| 传 Office 文件返回 `503` | Ubuntu **里面**没装 LibreOffice | `apt-get install -y libreoffice` |

## 直接在 Windows 上装行不行

目前不支持。这个组合没有 CI 覆盖，也没有端到端验证过。
