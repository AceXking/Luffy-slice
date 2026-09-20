"""本地配置：API 密钥等。

说明：
- 这里存放的密钥仅用于本地/私有部署；若仓库要提交到公开远端，
  建议把本文件加入 .gitignore，改用环境变量 PEXELS_API_KEY / PIXABAY_API_KEY 注入。
- bg_provider 与 app 会优先用环境变量，环境变量缺失时回退到本文件常量。
"""

# Pexels 视频 API key（https://www.pexels.com/api/ 免费申请）
PEXELS_API_KEY = "Ae6AvdeE2FcQb7IH7PVi0QnytUbILiiemVY8f5qqyWSHIBy42Dk4IaXX"

# Pixabay 视频 API key（https://pixabay.com/api/docs/ 免费申请），留空则不使用
PIXABAY_API_KEY = ""
