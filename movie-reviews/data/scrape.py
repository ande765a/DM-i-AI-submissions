import requests
import re

if __name__ == "__main__":
	page = 20

	while True:
		try:
			url = f"https://www.rottentomatoes.com/api/private/v2.0/browse?maxTomato=100&services=amazon%3Bhbo_go%3Bitunes%3Bnetflix_iw%3Bvudu%3Bamazon_prime%3Bfandango_now&certified&sortBy=release&type=dvd-streaming-all&page={page}"

			response = requests.get(url)
			data = response.json()
			
			for result in data["results"]:
				url = result["url"]
				html = requests.get(f"https://www.rottentomatoes.com/{url}").text
				titleId = re.search(r"\"titleId\":\"([a-zA-Z0-9-]+)\"", html).group(1)
				
				review_url = f"https://www.rottentomatoes.com/napi/movie/{titleId}/reviews/user?direction=next"
				review_response = requests.get(review_url).json()

				for review in review_response["reviews"]:
					text = review["review"].replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
					rating = review["rating"]
					#review = review["review"]
					print(f"{url}\t{titleId}\t{rating}\t{text}")

				#print(review_response)
				
			
			page += 1
		except Exception as e:
			print(e)
			print("Stopping...")
			break


		"""
		https://www.rottentomatoes.com/napi/movie/f0332c7a-b0aa-3c80-a064-ab3e2eb643ca/reviews/user?direction=next

		
		
		https://www.rottentomatoes.com/napi/movie/c7932403-fa74-3121-82d4-2eeae2dd4d5f/reviews/user?direction=next
		"""

		

	
	






