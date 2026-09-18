# Design: macOS Native LLM Status Item

## Architecture

新增 `mac_status_item.py`，使用系统 Python 提供的 PyObjC (`AppKit`, `Foundation`) 创建 `NSStatusItem` 和下拉菜单。进程独立于 `gen.py` 和浏览器运行，通过 `NSTimer` 每秒读取一次模型状态；状态项进程只读项目状态文件和 `llm.log`，不持有模型进程，也不改变推理逻辑。Emoji 开关启用时始终保留状态项；没有模型或模型暂停时显示等待文案，关闭开关则退出状态项进程。

状态项标题采用短文本/符号以适应菜单栏宽度；下拉菜单显示完整状态。状态映射如下：

- 无模型或服务停止：`等待模型🥳`，点击后为 `他在干嘛？发呆等你启动模型~🥳`
- 模型进程已启动但健康检查未通过：`阅读🤨`，点击后为 `他在干嘛？阅读内容~🤨`
- `activity.state=prompt`：`阅读🤨`，点击后附加读取百分比
- `activity.state=gen`：`组织语言🥲`，点击后为 `他在干嘛？组织语言~🥲 · 已生成 {n} token`
- `activity.state=tool`：`使用工具🔧`，点击后为 `他在干嘛？尝试使用工具~🔧`
- `activity.state=idle`：`发呆😶`，点击后为 `他在干嘛？发呆中~😶`
- 无法读取状态或发生异常：`异常😶`

当状态项被点击时，菜单显示最近一次状态；后台定时器每秒自动更新，不提供手动刷新按钮。菜单仅包含“退出状态栏”命令。首版不提供启动/暂停/停止模型操作，避免绕过现有网页控制台的服务互斥逻辑。

## Lifecycle

`start.sh start` 在检测到 `uname` 为 `Darwin` 后，以 `nohup python3 mac_status_item.py` 启动独立守护进程，并写入 `mac_status_item.pid`；已存在且存活的 PID 不重复启动。语言模型通过网页或其他入口启动时，`llm.start_llm()` 也会幂等拉起同一守护进程，确保“只要后台有模型就有状态栏”。守护进程检测到语言模型运行后创建状态项。`start.sh stop` 读取该 PID，先发送 TERM，必要时在有限等待后结束残留进程并删除 PID 文件。非 macOS 平台不创建或启动该进程。

状态项进程启动时设置项目根目录为工作目录，并通过 `llm.py` 的公开 `stats()` 获取数据。若 PyObjC 不可用或状态项启动失败，应记录到 `mac_status_item.log` 并让 `start.sh` 继续启动网页，不影响现有功能。

## Interfaces

- `mac_status_item.py`: 无参数运行；可选 `--interval`（默认 1 秒）和 `--project-root`，便于测试。
- PID 文件：项目根目录 `mac_status_item.pid`。
- 日志文件：项目根目录 `mac_status_item.log`。
- 配置开关：`config.json` 的 `mac_status_item_enabled`，缺省为 `true`；`start.sh` 选项 3 可切换。
- 数据源：`llm.stats()` 返回的 `loading`、`model`、`activity` 字段。

## Compatibility and Failure Handling

- 仅在 `platform.system() == "Darwin"` 时导入 AppKit 并注册状态项；Windows/Linux 保持现状。
- 菜单栏状态项的位置由 macOS `NSStatusBar` 管理，保证出现在系统菜单栏右侧区域，但不承诺紧邻电池图标或插入系统图标内部。
- 读取状态失败时保留上次可用文本，并在菜单中显示“状态暂时不可用”；下次定时器继续重试。没有模型时显示等待状态项。
- 退出状态项不会停止语言模型；停止项目时由 `start.sh stop` 统一清理状态项。

## Verification Plan

- 纯函数测试：状态映射、百分比/ token 文本格式、异常与缺失字段处理。
- Shell 测试：macOS 条件启动幂等、PID 清理、非 macOS 跳过。
- macOS 手工验证：启动项目后关闭浏览器，确认菜单栏仍显示；模拟加载/读取/生成/空闲状态并点击菜单查看完整文本。

## Review

2026-09-02 人工设计评审通过。检查了状态映射的正反分支、缺失字段与状态读取失败、重复启动与停止清理、非 macOS 跳过、AppKit 主线程事件循环，以及状态项退出不影响语言模型等边界。保留一项平台限制：菜单栏相邻位置由 macOS 决定，验收只要求出现在右侧菜单栏，不要求紧贴电池图标。

## Verification

2026-09-02: `python3 -m unittest discover -s tests -q` passed all 38 tests; `python3 -m py_compile mac_status_item.py`, `zsh -n start.sh`, and `git diff --check` passed. AppKit/Foundation import and the status-item event-loop smoke check passed on macOS. Visual manual confirmation with the browser closed remains a user-side acceptance task because it requires observing the desktop menu bar.
