from datetime import datetime, timedelta
import json
import random
from openai import OpenAI
import inspect

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = "qwen2.5-7b-instruct-1m"

def call_function_with_json(func, json_data):
    
    # Lấy danh sách các tham số của hàm
    sig = inspect.signature(func)
    func_params = sig.parameters
    
    # Lọc và chuẩn bị các tham số từ JSON khớp với tham số của hàm
    filtered_params = {key: json_data.get(key, None) for key in func_params.keys()}
    
    # Gọi hàm với các tham số
    return func(**filtered_params)

def get_delivery_date(order_id: str) -> datetime:
    # Generate a random delivery date between today and 14 days from now
    # in a real-world scenario, this function would query a database or API
    today = datetime.now()
    random_days = random.randint(1, 14)
    delivery_date = today + timedelta(days=random_days)
    return delivery_date.strftime("%Y-%m-%d %H:%M:%S")

def search_product(product_name=None, category=None, range_price=None):
    products = [
        {
            "id": 1,
            "name": "Pate cho mèo",
            "price": 25000,
            "category_name": "Thức ăn cho mèo"
        },
        {
            "id": 2,
            "name": "Pate cho chó",
            "price": 25000,
            "category_name": "Thức ăn cho chó"
        },
        {
            "id": 3,
            "name": "Lông vũ đồ chơi cho mèo",
            "price": 25000,
            "category_name": "Đồ chơi cho mèo"
        }
    ]

    filtered_products = [
        product for product in products
        if (product_name is None or product_name.lower() in product["name"].lower()) and
           (category is None or category.lower() in product["category_name"].lower()) and
           (range_price is None or product["price"] <= range_price)
    ]
    
    return filtered_products

function_mapping = {
    "get_delivery_date": get_delivery_date,
    "search_product": search_product,
}


def process_tool_calls(response):
    # Extract tool call information from the response
    tool_calls = response.tool_calls

    if tool_calls:
        # Extract tool function details
        tool_call_id = tool_calls[0].id
        tool_function_name = tool_calls[0].function.name
        tool_query_string = eval(tool_calls[0].function.arguments)
        return tool_function_name, tool_query_string

    else:
        return None, None

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Nhận ngày giao hàng cho đơn hàng của khách hàng. Gọi đến đây bất cứ khi nào bạn cần biết ngày giao hàng, ví dụ khi khách hàng hỏi 'Gói hàng của tôi đâu'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "ID đơn hàng của khách hàng.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "Tìm kiếm sản phẩm dựa trên tên, danh mục hoặc mức giá tối đa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Tên của sản phẩm để tìm kiếm (có thể là một phần tên).",
                    },
                    "category": {
                        "type": "string",
                        "description": "Tên danh mục của sản phẩm để lọc, sử dụng tiếng Việt rõ ràng (ví dụ: 'Thức ăn cho mèo', 'Đồ chơi cho mèo').",
                    },
                    "range_price": {
                        "type": "number",
                        "description": "Giá tối đa của sản phẩm để lọc.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    
]

messages = [
    {
        "role": "system",
        "content": "Bạn là trợ lý hỗ trợ khách hàng. Mọi câu trả lời phải bằng tiếng Việt. Sử dụng công cụ được cung cấp để hỗ trợ người dùng.",
    },
    {
        "role": "user",
        "content": "Tôi muốn mua pate cho mèo vị cá ngừ",
    },
]

# LM Studio
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
)

#TODO Check if LLM using function_calling
if response.choices[0].finish_reason == 'tool_calls': 
    function_tool_name, query_data = process_tool_calls(response=response.choices[0].message)
    
    function_to_do = function_mapping[function_tool_name]
    result_function_called = call_function_with_json(func=function_to_do, json_data=query_data)
    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)

    assistant_tool_call_request_message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": response.choices[0].message.tool_calls[0].id,
                "type": response.choices[0].message.tool_calls[0].type,
                "function": response.choices[0].message.tool_calls[0].function,
            }
        ],
    }

    # Create a message containing the result of the function call
    function_call_result_message = {
        "role": "tool",
        "content": result_function_called,
        "tool_call_id": response.choices[0].message.tool_calls[0].id,
    }


    # Prepare the chat completion call payload
    completion_messages_payload = [
        messages[0],
        messages[1],
        assistant_tool_call_request_message,
        function_call_result_message,
    ]
    

    # Call the OpenAI API's chat completions endpoint to send the tool call result back to the model
    # LM Studio
    response = client.chat.completions.create(
        model=model,
        messages=completion_messages_payload,
    )

    print(response.choices[0].message.content, flush=True)

else:
    response = client.chat.completions.create(
        model=model,
        messages=[
            messages[0],
            messages[1],
            {
                "role": "tool",
                "content": "Không tìm thấy dữ liệu để trả lời" ,
                "tool_call_id": "405959860"
            }
        ],
    )

    print(response.choices[0].message.content, flush=True)