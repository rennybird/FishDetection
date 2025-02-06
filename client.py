import flet as ft
import requests
import time

def main(page: ft.Page):
    page.title = "Fish Count Dashboard"
    page.bgcolor = "#222222"
    
    count_text = ft.Text("Loading...", size=24, color="white")
    page.add(count_text)
    
    def update_counts():
        while True:
            try:
                response = requests.get("http://127.0.0.1:5000/fish_counts")
                data = response.json()
                
                count_text.value = f"Left: {data['left']}\nRight: {data['right']}"
                page.update()
            except:
                count_text.value = "Error fetching data!"
                page.update()
            
            time.sleep(2)  # Update every 2 seconds
    
    page.run_thread(update_counts)

ft.app(target=main, view=ft.WEB_BROWSER)
