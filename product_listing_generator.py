"""
AI Product Listing Generator
Ironhack - API Calling to ChatGPT Lab

Generates e-commerce product listings using OpenAI Vision.
Includes prompt optimization, quality scoring, and cost tracking.

API requests are only made when generation functions are called.
"""

import os
import json
import base64
import time
import numpy as np
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")

# API pricing (USD per 1 million tokens)
INPUT_PRICE = 0.15
CACHED_INPUT_PRICE = 0.075
OUTPUT_PRICE = 0.60

def get_openai_client():
    """Initialize the OpenAI client only when needed."""
    return OpenAI()

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""

    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")

    return encoded

def create_product_listing_prompt(product_name, price, category, additional_info=None):
    """
    Create a structured prompt for AI-generated product listings.
    """

    prompt = f"""
You are an expert e-commerce copywriter.

Analyze the provided product image and metadata to create
a professional, engaging, and SEO-friendly product listing.

PRODUCT INFORMATION:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{f'- Additional Information: {additional_info}' if additional_info else ''}

REQUIREMENTS:

1. PRODUCT TITLE
- Catchy and SEO-friendly
- Maximum 60 characters
- Include the product type and important attributes

2. PRODUCT DESCRIPTION
- 150-200 words
- Highlight key features and benefits
- Use professional and persuasive language
- Describe visible design elements and colors
- Avoid unsupported claims

3. KEY FEATURES
- Provide 5-7 bullet points
- Focus on relevant product characteristics
- Use clear and concise language

4. SEO KEYWORDS
- Provide 10-15 relevant keywords
- Include product type, category, and relevant attributes

IMPORTANT RULES:
- Use the product image as the primary visual reference.
- Do not invent materials, specifications, or technical features.
- Only mention attributes supported by the image or metadata.
- Return valid JSON only, without markdown formatting.

OUTPUT FORMAT:

{{
    "title": "Product title here",
    "description": "Full product description here",
    "features": ["Feature 1", "Feature 2"],
    "keywords": "keyword1, keyword2, keyword3"
}}
"""

    return prompt

def estimate_batch_cost(number_of_products, cost_per_request, regeneration_rate=0):
    """
    Estimate API costs for processing multiple products.

    Assumes each regenerated product requires one extra API call.
    """

    initial_cost = number_of_products * cost_per_request

    regeneration_cost = initial_cost * regeneration_rate

    total_estimated_cost = initial_cost + regeneration_cost

    return {
        "products": number_of_products,
        "initial_cost": round(initial_cost, 4),
        "regeneration_cost": round(regeneration_cost, 4),
        "total_cost": round(total_estimated_cost, 4)
    }

def create_optimized_prompt(product_name, price, category, additional_info=None):

    return f"""
Create an accurate, SEO-friendly e-commerce listing using
the product image and metadata.

Product: {product_name}
Price: ${price:.2f}
Category: {category}
Additional info: {additional_info or 'None'}

Return valid JSON with:
- title: SEO-friendly, max 60 characters
- description: 150-200 words, persuasive and factual
- features: array of 5-7 concise features
- keywords: string of 10-15 comma-separated SEO keywords

Describe visible attributes accurately.
Do not invent materials, specifications, or unsupported claims.

JSON fields: title, description, features, keywords.
Return JSON only.
"""

def test_prompt_version(prompt, image_base64):
    client = get_openai_client()

    result = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            }
        ],
        store=False
    )

    input_tokens = result.usage.input_tokens
    output_tokens = result.usage.output_tokens

    cached_tokens = (
        result.usage.input_tokens_details.cached_tokens
        if result.usage.input_tokens_details else 0
    ) or 0

    cost = (
        (input_tokens - cached_tokens) * INPUT_PRICE
        + cached_tokens * CACHED_INPUT_PRICE
        + output_tokens * OUTPUT_PRICE
    ) / 1_000_000

    # Preserve the raw response before parsing
    raw_output = result.output_text

    raw_output = raw_output.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.split("\n", 1)[1]
        raw_output = raw_output.rsplit("```", 1)[0].strip()

    try:
        listing = json.loads(raw_output)
        json_valid = True
        error = None

    except json.JSONDecodeError as e:
        listing = None
        json_valid = False
        error = str(e)

    return {
        "listing": listing,
        "raw_output": raw_output,
        "json_valid": json_valid,
        "error": error,
        "response_status": result.status,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": result.usage.total_tokens,
        "cost": cost
    }

def score_listing(listing):

    score = 0
    issues = []

    required_fields = [
        "title",
        "description",
        "features",
        "keywords"
    ]

    # 1. Required fields — 20 points
    if all(field in listing for field in required_fields):
        score += 20
    else:
        issues.append("Missing required fields")

    # 2. Title length — 20 points
    title = listing.get("title", "")

    if isinstance(title, str) and 0 < len(title) <= 60:
        score += 20
    else:
        issues.append("Invalid title length")

    # 3. Description length — 25 points
    description = listing.get("description", "")
    word_count = (
        len(description.split())
        if isinstance(description, str) else 0
    )

    if 150 <= word_count <= 200:
        score += 25
    else:
        issues.append(f"Description length: {word_count} words")

    # 4. Features count — 20 points
    features = listing.get("features", [])

    if isinstance(features, list) and 5 <= len(features) <= 7:
        score += 20
    else:
        issues.append("Invalid features count")

    # 5. Keywords count — 15 points
    keywords = listing.get("keywords", "")

    keyword_count = (
        len([k for k in keywords.split(",") if k.strip()])
        if isinstance(keywords, str) else 0
    )

    if 10 <= keyword_count <= 15:
        score += 15
    else:
        issues.append(f"Keywords count: {keyword_count}")

    return {
        "score": score,
        "passed": score >= 80,
        "issues": issues
    }

def regenerate_if_needed(listing, max_retries=1):

    updated_listing = listing.copy()
    total_regeneration_cost = 0
    regeneration_input_tokens = 0
    regeneration_output_tokens = 0
    attempts = 0

    quality = score_listing(updated_listing)

    while not quality["passed"] and attempts < max_retries:

        # Only handle short descriptions in this exercise
        if not any(
            issue.startswith("Description length:")
            for issue in quality["issues"]
        ):
            break

        prompt = f"""
Rewrite the following e-commerce product description.

Product title: {updated_listing['title']}

Current description:
{updated_listing['description']}

Requirements:
- Target 175-185 words. Do not write fewer than 160 words.
- Preserve the original product information.
- Use clear, persuasive e-commerce language.
- Do not invent materials, specifications, or product claims.
- Return only the rewritten description as plain text.
"""
        client = get_openai_client()
        result = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            store=False
        )

        # Track API cost
        usage = result.usage
        regeneration_input_tokens += usage.input_tokens
        regeneration_output_tokens += usage.output_tokens

        cached_tokens = (
            usage.input_tokens_details.cached_tokens
            if usage.input_tokens_details else 0
        ) or 0

        request_cost = (
            (usage.input_tokens - cached_tokens) * INPUT_PRICE
            + cached_tokens * CACHED_INPUT_PRICE
            + usage.output_tokens * OUTPUT_PRICE
        ) / 1_000_000

        total_regeneration_cost += request_cost
        attempts += 1

        # Replace only the description
        if result.output_text.strip():
            updated_listing["description"] = result.output_text.strip()

        # Re-evaluate quality
        quality = score_listing(updated_listing)

    return {
        "listing": updated_listing,
        "quality": quality,
        "attempts": attempts,
        "regeneration_cost": total_regeneration_cost,
        "input_tokens": regeneration_input_tokens,
        "output_tokens": regeneration_output_tokens,
    }

def process_product(product):

    # 1. Generate prompt
    prompt = create_optimized_prompt(
        product_name=product["name"],
        price=product["price"],
        category=product["category"],
        additional_info=(
            f"Color: {product['color']}, "
            f"Subcategory: {product['subcategory']}"
        )
    )

    # 2. Encode image
    image_base64 = encode_image_to_base64(product["image_path"])

    # 3. Generate listing and track initial API cost
    generation = test_prompt_version(prompt, image_base64)

    # 4. Validate JSON
    if not generation["json_valid"]:
        return {
            "product_id": product["id"],
            "status": "JSON_ERROR",
            "quality_score": 0,
            "api_cost": generation["cost"],
            "error": generation["error"]
        }

    listing = generation["listing"]

    # 5. Evaluate quality
    initial_quality = score_listing(listing)

    # 6. Regenerate if necessary
    regeneration = regenerate_if_needed(
        listing,
        max_retries=1
    )

    # 7. Calculate total API cost
    total_api_cost = (
        generation["cost"]
        + regeneration["regeneration_cost"]
    )

    # 8. Return complete result
    return {
        "product_id": product["id"],
        "product_name": product["name"],
        "listing": regeneration["listing"],
        "initial_quality_score": initial_quality["score"],
        "final_quality_score": regeneration["quality"]["score"],
        "quality_status": (
            "APPROVED"
            if regeneration["quality"]["passed"]
            else "REVIEW_REQUIRED"
        ),
        "regeneration_attempts": regeneration["attempts"],
        "input_tokens": (
            generation["input_tokens"]
            + regeneration["input_tokens"]
        ),

        "output_tokens": (
            generation["output_tokens"]
            + regeneration["output_tokens"]
        ),

        "total_tokens": (
            generation["total_tokens"]
            + regeneration["input_tokens"]
            + regeneration["output_tokens"]
        ),
        "total_api_cost": total_api_cost
    }

def json_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Product Listing Generator"
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Explicitly enable OpenAI API calls"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of products to process"
    )

    parser.add_argument(
        "--output",
        default="batch_generated_listings.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    if not args.generate:
        print("DRY RUN: No API calls will be made.")
        print("Use --generate to explicitly enable generation.")
        return

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    output_path = Path(args.output)

    if output_path.exists():
        parser.error(
            f"Output file already exists: {output_path}. "
            "Choose another filename to avoid overwriting results."
        )

    products = pd.read_csv("products.csv")
    selected_products = products.head(args.limit)

    print(f"Selected products: {len(selected_products)}")
    print("OpenAI API calls are enabled.")

    results = []

    for _, product in selected_products.iterrows():
        print(f"Processing: {product['name']}")
        result = process_product(product)
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_converter
        )

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
    