# import requests

# url = "https://api.football-data.org/v4/competitions/PL/standings"
# headers = {"X-Auth-Token": "ef9ca5ca695b48ac9598a6d08fd8c495"}

# response = requests.get(url, headers=headers)

# if response.status_code == 200:
#     data = response.json()
#     standings = data["standings"][0]["table"]

#     print("🏆 BẢNG XẾP HẠNG NGOẠI HẠNG ANH 🏆\n")
#     for team in standings:
#         print(f"{team['position']}. {team['team']['name']} - {team['points']} điểm")
#         print(f"   - Trận đã đấu: {team['playedGames']}")
#         print(f"   - Thắng: {team['won']}, Hòa: {team['draw']}, Thua: {team['lost']}")
#         print(f"   - Hiệu số: {team['goalDifference']}")
#         print("----------------------------------")
# else:
#     print("Lỗi:", response.status_code, response.text)

# lấy top ghi bàn

# import requests

# # URL API để lấy danh sách cầu thủ ghi bàn hàng đầu của Premier League
# url = "https://api.football-data.org/v4/competitions/PL/scorers"

# # Thay thế bằng token API của bạn
# headers = {"X-Auth-Token": "ef9ca5ca695b48ac9598a6d08fd8c495"}

# # Gửi yêu cầu GET đến API
# response = requests.get(url, headers=headers)

# # Kiểm tra nếu yêu cầu thành công
# if response.status_code == 200:
#     data = response.json()
#     scorers = data["scorers"]

#     print("⚽ TOP GHI BÀN NGOẠI HẠNG ANH ⚽\n")
#     for player in scorers:
#         print(f"{player['player']['name']} ({player['team']['name']}) - {player['goals']} bàn thắng")
#         print(f"   - Số lần kiến tạo: {player.get('assists', 0)}")  # Một số cầu thủ có thể không có thông tin kiến tạo
#         print(f"   - Số trận đã chơi: {player['playedMatches']}")
#         print("----------------------------------")
# else:
#     print("Lỗi:", response.status_code, response.text)


#lấy tin tức mới nhất:
import requests

API_KEY = "e9fc6b8a56814e6da8b0bbc373f84375"  # Thay bằng API Key của bạn
url = f"https://newsapi.org/v2/everything?q=Premier+League&language=vi&sortBy=publishedAt&apiKey={API_KEY}"

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    articles = data["articles"][:5]  # Lấy 5 tin mới nhất

    print("📰 TIN TỨC BÓNG ĐÁ ANH MỚI NHẤT 📰\n")
    for article in articles:
        print(f"📌 {article['title']}")
        print(f"🗞 Nguồn: {article['source']['name']}")
        print(f"📅 Ngày: {article['publishedAt']}")
        print(f"🔗 Link: {article['url']}\n")
        print(f"📝 Mô tả: {article['description']}\n")
        print("-" * 50)
else:
    print("Lỗi khi lấy tin tức:", response.status_code)

