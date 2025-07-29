from serpapi import GoogleSearch

params = {
    "engine": "google_shopping",
    "q": "Acer monitor",
    "api_key": "84c239e80e11a82f56b168fe0e75ddbfc0e4b1c85a968aa42c9e9b5e6a69868c"
}

search = GoogleSearch(params)
results = search.get_dict()
shopping_results = results.get("shopping_results", [])
print("Shopping Results:", shopping_results)

# Print any error message from SerpApi
if 'error' in results:
    print("SerpApi Error:", results['error']) 