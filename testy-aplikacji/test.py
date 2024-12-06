import requests
import time
import os
import csv
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://167.99.241.72:8051/predict/"

IMAGE_DIR = "./test_images"
OUTPUT_FILE = "test_results.csv"
AVERAGE_FILE = "test_averages.csv"

def test_single_image(image_path):
    with open(image_path, "rb") as image_file:
        files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}
        start_time = time.time()
        response = requests.post(API_URL, files=files)
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "image": os.path.basename(image_path),
                "status_code": response.status_code,
                "prediction": data.get("prediction", ""),
                "confidence": data.get("confidence", ""),
                "receiving_time": data["times"].get("receiving", ""),
                "preprocessing_time": data["times"].get("preprocessing", ""),
                "prediction_time": data["times"].get("prediction", ""),
                "total_time": data["times"].get("total", ""),
            }
        else:
            return {
                "image": os.path.basename(image_path),
                "status_code": response.status_code,
                "prediction": "N/A",
                "confidence": "N/A",
                "receiving_time": "N/A",
                "preprocessing_time": "N/A",
                "prediction_time": "N/A",
                "total_time": total_time,
            }

def test_multiple_images(image_files, threads):
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(test_single_image, img) for img in image_files]
        results = [future.result() for future in futures]
    return results

def save_results_to_csv(results, mode):
    with open(OUTPUT_FILE, mode, newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        if mode == "w": 
            csvwriter.writerow([
                "Image", "Status Code", "Prediction", "Confidence",
                "Receiving Time (s)", "Preprocessing Time (s)", 
                "Prediction Time (s)", "Total Time (s)", "Mode"
            ])
        for result in results:
            csvwriter.writerow([
                result["image"], result["status_code"], result["prediction"],
                result["confidence"], result["receiving_time"],
                result["preprocessing_time"], result["prediction_time"],
                result["total_time"], result.get("mode", "single")
            ])

def save_averages_to_csv(averages, overall_average):
    with open(AVERAGE_FILE, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            "Image", "Mode", "Average Receiving Time (s)", "Average Preprocessing Time (s)",
            "Average Prediction Time (s)", "Average Total Time (s)"
        ])
        for avg in averages:
            csvwriter.writerow([
                avg["image"], avg["mode"], avg["avg_receiving_time"],
                avg["avg_preprocessing_time"], avg["avg_prediction_time"],
                avg["avg_total_time"]
            ])
        # Write overall average
        csvwriter.writerow([])
        csvwriter.writerow(["Overall Averages"])
        csvwriter.writerow(["Mode", "Average Total Time (s)"])
        for mode, avg_total_time in overall_average.items():
            csvwriter.writerow([mode, avg_total_time])

def calculate_averages(results):
    averages = {}
    for result in results:
        key = (result["image"], result["mode"])
        if key not in averages:
            averages[key] = {
                "image": result["image"],
                "mode": result["mode"],
                "receiving_times": [],
                "preprocessing_times": [],
                "prediction_times": [],
                "total_times": []
            }
        averages[key]["receiving_times"].append(float(result["receiving_time"]))
        averages[key]["preprocessing_times"].append(float(result["preprocessing_time"]))
        averages[key]["prediction_times"].append(float(result["prediction_time"]))
        averages[key]["total_times"].append(float(result["total_time"]))

    averages_list = []
    for key, values in averages.items():
        averages_list.append({
            "image": values["image"],
            "mode": values["mode"],
            "avg_receiving_time": sum(values["receiving_times"]) / len(values["receiving_times"]),
            "avg_preprocessing_time": sum(values["preprocessing_times"]) / len(values["preprocessing_times"]),
            "avg_prediction_time": sum(values["prediction_times"]) / len(values["prediction_times"]),
            "avg_total_time": sum(values["total_times"]) / len(values["total_times"]),
        })

    overall_average = {}
    mode_times = {}
    for avg in averages_list:
        mode = avg["mode"]
        if mode not in mode_times:
            mode_times[mode] = []
        mode_times[mode].append(avg["avg_total_time"])
    for mode, times in mode_times.items():
        overall_average[mode] = sum(times) / len(times)

    return averages_list, overall_average

def main():
    image_files = [os.path.join(IMAGE_DIR, img) for img in os.listdir(IMAGE_DIR) if img.endswith((".jpg", ".jpeg", ".png"))]
    
    if len(image_files) < 10:
        print("W katalogu powinno znajdować się co najmniej 10 zdjęć.")
        return

    image_files = image_files[:10]

    print("Rozpoczynam test jednowątkowy...")
    single_thread_results = []
    for i, image_path in enumerate(image_files):
        print(f"Przetwarzanie zdjęcia {i + 1}/{len(image_files)}: {os.path.basename(image_path)}")
        for j in range(10):
            print(f"  Iteracja {j + 1}")
            result = test_single_image(image_path)
            result["mode"] = "single"
            single_thread_results.append(result)
    
    save_results_to_csv(single_thread_results, mode="w")
    print("Zapisano wyniki testów jednowątkowych do pliku.")

    print("Rozpoczynam test wielowątkowy...")
    multithread_results = test_multiple_images(image_files * 10, threads=10) 
    for result in multithread_results:
        result["mode"] = "multithreaded"
    
    save_results_to_csv(multithread_results, mode="a")
    print("Zapisano wyniki testów wielowątkowych do pliku.")

    all_results = single_thread_results + multithread_results
    averages, overall_average = calculate_averages(all_results)
    save_averages_to_csv(averages, overall_average)
    print("Zapisano średnie do pliku.")

if __name__ == "__main__":
    main()