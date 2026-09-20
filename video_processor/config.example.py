"""本地配置模板（复制为 config.py 后填写，config.py 已被 .gitignore 忽略）。

说明：
- 这里存放的密钥仅用于本地/私有部署；请勿把真实密钥提交到公开远端，
  建议改用环境变量 PEXELS_API_KEY / PIXABAY_API_KEY 注入。
- bg_provider 与 app 会优先用环境变量，环境变量缺失时回退到本文件常量。
"""

# Pexels 视频 API key（https://www.pexels.com/api/ 免费申请），留空则不使用
PEXELS_API_KEY = ""

# Pixabay 视频 API key（https://pixabay.com/api/docs/ 免费申请），留空则不使用
PIXABAY_API_KEY = ""
