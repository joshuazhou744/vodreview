from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
import os
import glob

def scrape_images(prompt, max_images=100, output_dir=None, engine="bing"):
    if not prompt:
        raise ValueError("prompt cannot be empty")
    
    if output_dir is None:
        safe_name = prompt.replace(" ", "_").lower()
        output_dir = os.path.join("datasets", safe_name, "images")

    os.makedirs(output_dir, exist_ok=True)
    if engine == "bing":
        crawler = BingImageCrawler(storage={"root_dir": output_dir})
    else:
        crawler = GoogleImageCrawler(storage={"root_dir": output_dir})

    crawler.crawl(keyword=prompt, max_num=max_images)

    paths = sorted(glob.glob(os.path.join(output_dir, "*")))
    print(f"[scrape] downloaded {len(paths)} images to {output_dir}")
    return paths

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    scrape_images(args.prompt, args.max_images, args.output_dir)