# AI Product Listing Generator — Lab Report

## 1. Project Overview

This project automates the creation of e-commerce product
listings using OpenAI's multimodal API.

The system combines product metadata and images to generate
structured product titles, descriptions, features, and keywords.

## 2. API Integration

- Model: gpt-4o-mini
- API: OpenAI Responses API
- Input: Product metadata and Base64-encoded images
- Output: Structured JSON product listings
- Authentication: API key loaded from environment variables

The implementation includes JSON parsing, error handling,
token usage tracking, and estimated API cost calculations.

## 3. Implementation

The dataset contains 20 products with corresponding images.

The project successfully generated listings for 10 products.

The workflow includes:

1. Loading product metadata and images.
2. Encoding images into Base64.
3. Constructing optimized prompts.
4. Sending multimodal requests to OpenAI.
5. Parsing and validating JSON responses.
6. Evaluating listing quality.
7. Regenerating descriptions when necessary.
8. Tracking token usage and estimated API costs.
9. Exporting generated listings to JSON and CSV.

## 4. Quality Evaluation

A rule-based quality scoring system evaluates:

- Required fields
- Title length
- Description length
- Number of product features
- Number of keywords

Listings that do not meet the quality threshold
can undergo one additional regeneration attempt.

The scoring system checks structural quality rather
than independently verifying factual accuracy.

## 5. Challenges and Solutions

**JSON formatting:** Model responses may contain Markdown
formatting or invalid JSON. The implementation removes
Markdown code fences and handles JSON parsing errors.

**Description length:** Some generated descriptions were
shorter than required. A targeted regeneration mechanism
was implemented.

**Cost control:** Token usage and estimated API costs
were tracked. The final Python script requires an explicit
--generate flag to initiate API requests.

**Reproducibility:** The dataset, images, Python script,
Jupyter Notebook, and generated results are included.

## 6. Results

- Dataset size: 20 products
- Generated listings: 10 products
- Unique product IDs: 10
- Missing required listing fields: 0
- JSON validation: Passed

## 7. Future Improvements

- Semantic and factual quality evaluation
- Automated detection of unsupported product claims
- More robust API retry and rate-limit handling
- Improved batch processing and checkpoint recovery
- Comparison of multiple models and prompt strategies

## 8. Deliverables

- product_listing_generator.py
- product_listing_generator.ipynb
- generated_listings.json
- products.csv
- product_images/
- SUBMISSION_REPORT.md