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
    
    if len(filtered_products) == 0:
        return "Không tìm thấy sản phẩm nào phù hợp với yêu cầu của bạn"
    json_string = json.dumps(filtered_products, ensure_ascii=False, indent=4)

    return json_string


def search_pet_info(question: None):
    return "Không tìm thấy dữ liệu câu trả lời cho câu hỏi trên."

function_mapping = {
    "get_delivery_date": get_delivery_date,
    "search_product": search_product,
    "search_pet_info": search_pet_info
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

question_description = """
                    Câu hỏi về chó mèo, có thể liên quan đến ngoại hình, đặc điểm, sức khỏe, bệnh tật, thức ăn hoặc giá bán. 
                    Câu hỏi phải được tóm tắt ngắn gọn, giữ nguyên các ý chính mà không làm mất thông tin quan trọng. 
                    Ví dụ:\n
                    - Câu hỏi ban đầu: 'Chó Corgi là một giống chó nhỏ, thông minh và dễ thương. Chúng rất thích hợp để nuôi trong căn hộ, vì kích thước nhỏ và khả năng giao tiếp tốt với con người. Giá của chúng thường dao động từ 5 đến 10 triệu đồng, tùy thuộc vào nguồn gốc và chất lượng.'\n
                    - Câu hỏi tóm tắt: 'Chó Corgi có đặc điểm gì và giá bao nhiêu?'\n
                    - Câu hỏi ban đầu: 'Để đảm bảo dinh dưỡng, thức ăn cho chó cần được lựa chọn kỹ lưỡng. Một chế độ ăn đa dạng, bao gồm đầy đủ chất đạm, chất béo và vitamin, là rất quan trọng. Thức ăn công nghiệp hiện nay đã đáp ứng được phần lớn các yêu cầu này.'\n
"""

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
    {
        "type": "function",
        "function": {
            "name": "search_pet_info",
            "description": "Tìm kiếm thông tin liên quan đến chó mèo dựa trên chủ đề câu hỏi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": question_description
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            }
        }
    }
    
]

messages = [
    {
        "role": "system",
        "sytem_message": "Only respond in Vietnamese",
        "content": "Bạn là trợ lý hỗ trợ khách hàng. Mọi câu trả lời phải bằng tiếng Việt. Sử dụng công cụ được cung cấp để hỗ trợ người dùng.",
    },
    {
        "role": "user",
        "content": "Đà Nẵng nằm ở đâu tại Việt Nam",
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
    print(function_tool_name)
    print(query_data)


    function_to_do = function_mapping[function_tool_name]
    result_function_called = call_function_with_json(func=function_to_do, json_data=query_data)
    print(result_function_called)

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

    if result_function_called == 'Không tìm thấy dữ liệu câu trả lời cho câu hỏi trên.':
        completion_messages_payload.append({
        "role": "assistant",
        "content": "Xin lỗi, hiện tại tôi không có dữ liệu để trả lời câu hỏi này."
    })
        

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
                "tool_call_id": "405959860",
                "sytem_message": "Only respond in Vietnamese"
            }
        ],
    )

    print(response.choices[0].message.content, flush=True)