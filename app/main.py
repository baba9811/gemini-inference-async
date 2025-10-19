import argparse
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import google.generativeai as genai
from PIL import Image
from io import BytesIO

from dataloader import load_fence_dataset

load_dotenv()

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env file")

genai.configure(api_key=GOOGLE_API_KEY)


class GeminiInference:
    """Gemini Async Inference Class"""

    def __init__(self, model_name: str = "gemini-2.5-flash", max_concurrent: int = 10):
        """
        Args:
            model_name: Gemini model name (gemini-2.0-flash-exp, gemini-1.5-pro, etc.)
            max_concurrent: Maximum concurrent requests
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def infer_single(self, text: str, image_data: Any = None, row_id: str = None) -> Dict[str, Any]:
        """
        Single sample async inference

        Args:
            text: Input text
            image_data: Image data (can be path string, bytes dict, or PIL Image)
            row_id: Data ID

        Returns:
            Inference result dictionary
        """
        async with self.semaphore:
            try:
                contents = []

                # Add image if available
                if image_data is not None:
                    image = None

                    # Handle different image data types
                    if isinstance(image_data, dict) and 'bytes' in image_data:
                        # HuggingFace dataset format: {'bytes': b'...'}
                        image = Image.open(BytesIO(image_data['bytes']))
                    elif isinstance(image_data, str) and os.path.exists(image_data):
                        # File path
                        image = Image.open(image_data)
                    elif isinstance(image_data, bytes):
                        # Raw bytes
                        image = Image.open(BytesIO(image_data))
                    elif isinstance(image_data, Image.Image):
                        # Already a PIL Image
                        image = image_data

                    if image is not None:
                        contents.append(image)

                # Add text
                if text:
                    contents.append(text)

                # Async generation
                response = await self.model.generate_content_async(contents)

                return {
                    "id": row_id,
                    "prediction": response.text,
                    "status": "success"
                }

            except Exception as e:
                return {
                    "id": row_id,
                    "prediction": None,
                    "status": "error",
                    "error_message": str(e)
                }

    async def infer_batch(
        self,
        df: pd.DataFrame,
        text_column: str,
        image_column: str = None
    ) -> List[Dict[str, Any]]:
        """
        Batch async inference

        Args:
            df: Input DataFrame
            text_column: Text column name
            image_column: Image data column name (optional)

        Returns:
            List of inference results
        """
        tasks = []
        for idx, row in df.iterrows():
            text = row.get(text_column, "")
            image_data = row.get(image_column, None) if image_column else None
            row_id = row.get("id", idx)

            task = self.infer_single(text, image_data, row_id)
            tasks.append(task)

        # Execute with tqdm progress bar
        results = await tqdm_asyncio.gather(*tasks, desc=f"Inferencing with {self.model_name}")

        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Gemini Multimodal Async Inference")

    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split (default: test)"
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default="input_query",
        help="Text input column name (default: input_query)"
    )

    parser.add_argument(
        "--image-column",
        type=str,
        default="image_path",
        help="Image path column name (default: image_path)"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="result",
        help="Output directory (default: result)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (default: None, all)"
    )

    return parser.parse_args()


async def main():
    """Main execution function"""
    args = parse_args()

    print(f"\n{'='*60}")
    print("Gemini Async Inference Started")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dataset Split: {args.split}")
    print(f"Text Column: {args.text_column}")
    print(f"Image Column: {args.image_column}")
    print(f"Max Concurrent: {args.max_concurrent}")
    print(f"{'='*60}\n")

    # Load dataset
    print("Loading dataset...")
    df = load_fence_dataset(split=args.split)

    # Limit samples if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {args.limit} samples")

    # Run inference
    print(f"\nStarting inference on {len(df)} samples...\n")
    inferencer = GeminiInference(model_name=args.model, max_concurrent=args.max_concurrent)
    results = await inferencer.infer_batch(df, args.text_column, args.image_column)

    # Convert results to DataFrame
    result_df = pd.DataFrame(results)

    # Create final DataFrame with only selected columns
    final_df = pd.DataFrame()
    final_df[args.text_column] = df[args.text_column]
    if args.image_column:
        final_df[args.image_column] = df[args.image_column]
    final_df["response"] = result_df["prediction"]
    final_df["status"] = result_df["status"]

    # Add error messages if any
    if "error_message" in result_df.columns:
        final_df["error_message"] = result_df["error_message"]

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Safe model name for filename
    safe_model_name = args.model.replace("/", "_").replace(":", "_")
    output_file = output_dir / f"{safe_model_name}_{args.split}_result.csv"

    final_df.to_csv(output_file, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("Inference Completed!")
    print(f"{'='*60}")
    print(f"Total samples: {len(final_df)}")
    print(f"Success: {(final_df['status'] == 'success').sum()}")
    print(f"Failed: {(final_df['status'] == 'error').sum()}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
