# API Calling to ChatGPT

> **How you'll submit this lab**
>
> This repo is your lab. Fork it, do the work described below in your fork, then open a pull
> request back into this repository. An AI reviewer will check your PR against `rubric.md` and
> leave feedback directly on the PR. See `README.md` for the full workflow.

**Scenario**  
You're working as an AI automation specialist for an e-commerce company. Your team needs to automate the creation of product listings from product images and basic information. Instead of manually writing descriptions, you'll use ChatGPT's vision capabilities to generate compelling product listings automatically. This will save hours of manual work and ensure consistent, high-quality product descriptions.

**Learning Objectives**
- [ ] Set up OpenAI API access and authentication
- [ ] Use ChatGPT API with vision capabilities (GPT-4 Vision)
- [ ] Send images and text prompts to the API
- [ ] Process API responses and handle errors
- [ ] Generate automated product listings from images and metadata
- [ ] Implement best practices for API usage and error handling

**Estimated Time:** 120-150 minutes

**Prerequisites:**
- [ ] Basic understanding of Python
- [ ] Familiarity with API concepts (HTTP requests)
- [ ] Understanding of JSON data format
- [ ] Knowledge of file handling in Python

---

## Introduction

Welcome to the ChatGPT API lab! In this exercise, you'll learn how to integrate OpenAI's ChatGPT API into your applications. Specifically, you'll use GPT-4 Vision to analyze product images and generate professional product listings.

**What you'll build:**
- An automated product listing generator
- Integration with OpenAI's ChatGPT API
- Image analysis using vision models
- Error handling and best practices

**Why this matters:**
APIs are the backbone of modern AI applications. Understanding how to call AI APIs, handle responses, and implement proper error handling is essential for building production-ready AI systems. This lab teaches you practical skills you'll use in real-world projects.

**Success criteria:**
- [ ] Successfully set up OpenAI API access
- [ ] Send images and text to ChatGPT API
- [ ] Generate product listings from images
- [ ] Handle API errors gracefully
- [ ] Process and format API responses

---

## Background Story

Your e-commerce company sells thousands of products, and creating compelling product listings is time-consuming. Each product needs:
- A catchy title
- A detailed description
- Key features highlighted
- SEO-friendly content

You've been tasked with building an automation system that:
1. Takes a product image and basic metadata (name, price, category)
2. Sends it to ChatGPT with a prompt
3. Receives a professionally written product listing
4. Formats it for use in your e-commerce platform

This will save your team hours of work and ensure consistent quality across all listings.

---

## Step-by-Step Instructions

### Step 1: Setting Up OpenAI API Access

**Objective:** Get API access and set up authentication. 

You have already received a key from Ironhack. Below are the steps to set your own personal key.

**What to do:**
1. Create an OpenAI account (if you don't have one)
2. Generate an API key
3. Install the OpenAI Python library
4. Set up secure API key storage

**Getting an API Key:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Click "Create new secret key"
5. Copy and save the key securely (you won't see it again!)

**Important Security Note:** Never commit API keys to version control! Always use environment variables or secure storage.

**Checkpoint:** Verify that your API key is set correctly and the client initializes without errors.

---

### Step 2: Preparing the Dataset

**Objective:** Download and set up the e-commerce product dataset.

**What to do:**
1. Download the e-commerce product dataset
2. Organize product images and metadata
3. Create a data structure for products

**Dataset Options:**

**Option 1: Products-10K Dataset (HuggingFace)**
- **Link:** [Fashion Product Images on HuggingFace](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
- Contains diverse product images
- Can be loaded directly with Python

**Option 2: Any Other Dataset**
- We recommend using a dataset with product images and metadata
- Ensure you have at least 10-20 products for testing
- Otherwise, you are free to use any dataset on Kaggle or Hugging Face

**Code template (using HuggingFace dataset):**

```python
# Install: pip install datasets
from datasets import load_dataset
import requests
from PIL import Image
import pandas as pd
from pathlib import Path

# Load dataset from HuggingFace
print("Loading product dataset...")
try:
    # Try loading the dataset
    dataset = load_dataset("ashraq/fashion-product-images-small", split="train[:100]")  # First 100 samples
    print(f"✓ Loaded {len(dataset)} products")
    
    # Convert to pandas for easier manipulation
    products_df = pd.DataFrame(dataset)
    print(f"Dataset columns: {products_df.columns.tolist()}")
    
except Exception as e:
    print(f"⚠ Could not load HuggingFace dataset: {e}")
    print("Using local images instead...")
    
    # Alternative: Use local images
    # Create a products.json file with product information
    products_data = [
        {
            "id": 1,
            "name": "Wireless Headphones",
            "price": 79.99,
            "category": "Electronics",
            "image_path": "images/product1.jpg"
        },
        # Add more products...
    ]
    
    products_df = pd.DataFrame(products_data)

# Create images directory
images_dir = Path("product_images")
images_dir.mkdir(exist_ok=True)

print(f"\n✓ Dataset prepared!")
print(f"  Total products: {len(products_df)}")
```

Tip: if you are not using the requests library, this code will need to be adapted.

Tip: if you are not using the PIL Image library, this code will need to be adapted.


**Expected outcome:** You should have product data loaded and ready to process.

**Checkpoint:** Verify that you can access product information and images.

---

### Step 3: Encoding Images for API

**Objective:** Convert product images to base64 format for API transmission.

**Code template:**
```python
import base64

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded

# Example usage
sample_path = products_df.iloc[0]["image_path"]
encoded_image = encode_image_to_base64(sample_path)
print(f"Encoded image length: {len(encoded_image)} characters")
print(f"Encoded prefix: {encoded_image[:40]}...")
```

**Expected outcome:** You should be able to encode images to base64 format.

**Checkpoint:** Verify that image encoding works correctly.

---

### Step 4: Creating the Product Listing Prompt

**Objective:** Design an effective prompt for generating product listings.

**What to do:**
1. Create a prompt template
2. Include product metadata
3. Specify the desired output format

**Code template:**
```python
def create_product_listing_prompt(product_name, price, category, additional_info=None):
    """
    Create a prompt for generating product listings.
    
    Parameters:
    - product_name: Name of the product
    - price: Price of the product
    - category: Product category
    - additional_info: Optional additional information
    
    Returns:
    - Formatted prompt string
    """
    prompt = f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{f'- Additional Info: {additional_info}' if additional_info else ''}

Please create a professional product listing that includes:

1. **Product Title** (catchy, SEO-friendly, 60 characters max)
2. **Product Description** (detailed, 150-200 words)
   - Highlight key features and benefits
   - Use persuasive language
   - Include relevant details visible in the image
3. **Key Features** (bullet points, 5-7 items)
4. **SEO Keywords** (comma-separated, 10-15 relevant keywords)

Format your response as JSON with the following structure:
{{
    "title": "Product title here",
    "description": "Full description here",
    "features": ["Feature 1", "Feature 2", ...],
    "keywords": "keyword1, keyword2, ..."
}}

Be specific about what you see in the image. Mention colors, materials, design elements, and any distinctive features."""
    
    return prompt

# Test prompt creation
test_prompt = create_product_listing_prompt(
    product_name="Wireless Bluetooth Headphones",
    price=79.99,
    category="Electronics",
    additional_info="Noise cancelling, 30-hour battery"
)

print("\n" + "="*50)
print("PROMPT TEMPLATE")
print("="*50)
print(test_prompt[:500] + "...")  # Show first 500 characters
```

**Expected outcome:** You should have a well-structured prompt template.

**Checkpoint:** Verify that your prompt includes all necessary information and clear instructions.

---

### Step 5: Calling the ChatGPT API with Vision

**Objective:** Send image and text to ChatGPT API and receive response.

**What to do:**
1. Prepare the API request with image and prompt
2. Call the ChatGPT API
3. Handle the response
4. Parse JSON output

**Expected outcome:** You should receive a JSON response with the generated product listing.

**Checkpoint:** Verify that:
- API call succeeds
- Response is received
- JSON is parsed correctly

---

### Step 6: Processing Multiple Products

**Objective:** Generate listings for multiple products in batch.

**What to do:**
1. Loop through products
2. Generate listing for each
3. Save results
4. Handle errors gracefully

**Expected outcome:** You should process multiple products and save all generated listings.

**Checkpoint:** Verify that:
- Multiple products are processed
- Results are saved correctly
- Errors are handled gracefully

---

## Submission Guidelines

### What to Submit

**Required deliverables:**
- [ ] Your complete `product_listing_generator.py` file
- [ ] Generated product listings (JSON file with results)
- [ ] Screenshots or output showing:
  - Successful API calls
  - Generated listings for at least 3 products
  - Error handling examples
- [ ] A brief report (1 page) including:
  - How the API integration works
  - Challenges you faced
  - Quality of generated listings
  - Potential improvements

### How to Submit

**Upload your work:**
1. **For code files:** Upload your Python script
2. **For JSON files:** Upload generated listings
3. **For screenshots:** Upload PNG or JPG images
4. **For reports:** Upload PDF or text file

**Important notes:**
- **DO NOT** include your API key in submitted files
- Make sure your code uses environment variables for the API key
- Test your code before submitting
- Include comments explaining your approach

**Due date:** Check with your instructor for the specific due date.

---

## Troubleshooting

**Common issues and solutions:**

**Issue 1: "Invalid API key" error**
- **Solution:** 
  - Verify your API key is correct
  - Make sure it's set as an environment variable
  - Check that you have credits in your OpenAI account
  - Ensure you're using the correct API key format

**Issue 2: "Rate limit exceeded" error**
- **Solution:** 
  - Add delays between API calls
  - Implement exponential backoff
  - Check your API tier limits
  - Reduce batch size

**Issue 3: "Image encoding fails"**
- **Solution:** 
  - Check image file exists and is readable
  - Verify image format is supported (JPEG, PNG)
  - Check file permissions
  - Ensure image isn't corrupted

**Issue 4: "JSON parsing error"**
- **Solution:** 
  - The API might return markdown-formatted JSON
  - Strip code block markers (```json and ```)
  - Add fallback to return text if JSON parsing fails
  - Check the raw response to see format

**Issue 5: "API response is slow"**
- **Solution:** 
  - Vision models are slower than text-only models
  - This is normal - be patient
  - Consider processing in parallel (advanced)
  - Reduce image size if possible

**Issue 6: "Generated listings are generic"**
- **Solution:** 
  - Improve your prompt with more specific instructions
  - Include more product metadata in the prompt
  - Adjust temperature parameter (higher = more creative)
  - Ask for more specific details about the image

---

## Bonus Challenges

If you finish early, try these additional challenges:

### Challenge 1: Multi-Image Analysis
Modify your code to analyze multiple product images:
- Send 2-3 images of the same product (different angles)
- Generate a listing that incorporates details from all images
- Compare results with single-image approach

### Challenge 2: Style Customization
Add style options to your prompts:
- Formal/casual tone
- Technical/simple language
- Different target audiences
- A/B test different styles (more on this in future modules!)

### Challenge 3: Cost Tracking
Implement cost tracking:
- Calculate tokens used per request
- Track total API costs
- Estimate costs for processing large batches
- Optimize prompts to reduce costs

### Challenge 4: Quality Scoring
Add quality metrics:
- Check if generated listing includes all required elements
- Score listings based on length, keyword density, etc.
- Flag low-quality listings for review
- Implement automatic re-generation for low scores

### Challenge 5: Integration with E-commerce Platform
Create a complete workflow:
- Export listings in platform-specific format (CSV, JSON)
- Include product IDs and SKUs
- Generate multiple listing variations
- Create a simple web interface for the tool

---

## Additional Resources

**Reference materials:**
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [GPT-4 Vision Guide](https://platform.openai.com/docs/guides/vision)
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [API Best Practices](https://platform.openai.com/docs/guides/production-best-practices)

**Tools and software:**
- [OpenAI Platform](https://platform.openai.com/) - API dashboard
- [Postman](https://www.postman.com/) - API testing tool
- [JSON Formatter](https://jsonformatter.org/) - Validate JSON responses

**Further reading:**
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [API Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [Error Handling Best Practices](https://platform.openai.com/docs/guides/error-codes)

---

## To-Do List

- [ ] Complete Step 1: Setting Up OpenAI API Access
- [ ] Complete Step 2: Preparing the Dataset
- [ ] Complete Step 3: Encoding Images for API
- [ ] Complete Step 4: Creating the Product Listing Prompt
- [ ] Complete Step 5: Calling the ChatGPT API with Vision
- [ ] Complete Step 6: Processing Multiple Products
- [ ] Complete Step 7: Error Handling and Best Practices
- [ ] Generate listings for at least 3 products
- [ ] Test error handling scenarios
- [ ] Review and improve your code
- [ ] Submit your work

---

## Learning Reflection

After completing this lab, reflect on what you've learned:

1. **API Integration:** What are the key steps in integrating an external API?
2. **Error Handling:** Why is robust error handling important for production systems?
3. **Prompt Engineering:** How did your prompt design affect the quality of generated listings?
4. **Image Processing:** What challenges did you face working with images in APIs?
5. **Automation:** How does this automation save time compared to manual work?
6. **Best Practices:** What security and efficiency practices did you implement?

**Key takeaways:**
- APIs enable powerful AI capabilities without building models from scratch
- Proper error handling and retry logic are essential for reliability
- Prompt engineering significantly affects output quality
- Security (API key management) is crucial
- Rate limiting and cost management are important considerations
- Automation can dramatically improve efficiency

These skills are directly applicable to building real-world AI applications and integrating AI services into production systems.

---

*Great work on building your automated product listing generator! You've learned essential skills for working with AI APIs that you'll use throughout your AI career.*
