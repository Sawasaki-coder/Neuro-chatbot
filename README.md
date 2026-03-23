# Neuro-chatbot 🤖

Dự án AI Chatbot được xây dựng dựa trên kiến trúc GPT-2 (Fine-tuned). Hệ thống bao gồm mã nguồn để huấn luyện mô hình với bộ dữ liệu Hỏi-Đáp tùy chỉnh và tích hợp trực tiếp vào Discord để tương tác theo thời gian thực.

---

## 🛠️ Yêu cầu hệ thống (Requirements)

Để chạy dự án này, máy tính của bạn cần cài đặt Python 3.8+ và các thư viện sau. 
Bạn có thể cài đặt nhanh bằng lệnh:
```bash
pip install torch torchvision torchaudio transformers discord.py pandas matplotlib
(Lưu ý: Khuyến nghị cài đặt phiên bản torch có hỗ trợ CUDA nếu bạn muốn chạy mô hình trên GPU để phản hồi nhanh hơn).
```
# 📦 Hướng dẫn Cài đặt & Tải Mô hình
Để tối ưu không gian lưu trữ và đảm bảo tốc độ tải code, trọng số của mô hình (Model Weights) được lưu trữ riêng trên Hugging Face. Vui lòng làm theo các bước sau để khởi chạy bot.

## Bước 1: Clone Repository
```bash
git clone [https://github.com/](https://github.com/)[TÊN_USER_CỦA_BẠN]/Neuro-chatbot.git
cd Neuro-chatbot
```
## Bước 2: Tải mô hình từ Hugging Face
Truy cập vào đường link Hugging Face của dự án:
👉 https://huggingface.co/nvnghiait/neuro-chatbot/tree/main

Tải toàn bộ các file cấu hình và trọng số (bao gồm vocab.json, merges.txt, config.json, và file model .safetensors hoặc .bin).

Đặt tất cả các file vừa tải vào một thư mục có tên là neuro_sama_model/ tại thư mục gốc của project.

## Bước 3: Cấu hình Discord Bot Token
Truy cập Discord Developer Portal.

Tạo một Application mới và thiết lập Bot.

Đảm bảo bạn đã bật Message Content Intent trong phần Bot settings.

Lấy Token của bot và thay thế vào biến DISCORD_TOKEN trong file chạy bot của bạn.

## Bước 4: Khởi chạy Bot
Chạy file python chứa mã nguồn Discord bot:

```bash
python run_bot.py
```

# 💡 Cách sử dụng trên Discord
Trong server: Nhập lệnh !chat <nội dung tin nhắn> (Ví dụ: !chat Xin chào Neuro!).

Nhắn tin trực tiếp (DM): Bạn có thể nhắn thẳng cho Bot mà không cần dùng tiền tố !chat.
