# djy

## API 概览
- `GET /`：返回服务信息和主要接口说明。
- `GET /conversation/files/{user_name}`：按时间倒序列出指定用户的对话文件 ID 及修改时间。
- `GET /users`：返回数据库中的用户姓名列表。
- `POST /chat`：核心对话接口，字段包括 `user_name`、`message`、可选的 `conversation_history`、`conversation_file_id`、`image_base64` 等。支持情感需求提取、健康类问题 RAG、提醒解析，实时将对话追加到对应文件。
- `POST /end`：结束对话并持久化，更新情感需求与健康日志；如有实时保存文件会复用，不再重复写入。
- `POST /reminder/notify`：提醒触发后的回调占位，当前直接返回成功。
- `GET /reminders/{user_name}`：读取并返回指定用户的提醒列表。

对话文件以 `history/{pinyin}_{userId}/conversation_时间戳.txt` 存储，单轮对话时即时追加，结束时会清理内存中的当前对话映射。情感需求与健康信息在对话过程中同步更新用户画像。
