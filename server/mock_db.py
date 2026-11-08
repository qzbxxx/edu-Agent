# mock_db.py
from datetime import datetime, timedelta


def build_mock_orders():
    """每次调用都重新生成一批订单，方便测试重置"""
    return {
        "ORD_001": {
            "order_id": "ORD_001",
            "user_id": "USER_A",
            "course_name": "Python 高级架构师训练营",
            "price": 1999.0,
            "pay_time": datetime.now() - timedelta(days=3),  # 3天前购买（符合退费政策）
            "learned_progress": 0.05,  # 学习进度 5%
            "status": "PAID"
        },
        "ORD_002": {
            "order_id": "ORD_002",
            "user_id": "USER_B",
            "course_name": "AIGC 商业化实战",
            "price": 2999.0,
            "pay_time": datetime.now() - timedelta(days=20),  # 20天前购买（超出7天无理由退款）
            "learned_progress": 0.60,
            "status": "PAID"
        }
    }


# 模拟订单数据库
MOCK_ORDERS = build_mock_orders()


def reset_mock_orders():
    """把订单状态恢复成初始值（测试用，避免上一条用例把订单退了导致下一条失败）"""
    MOCK_ORDERS.clear()
    MOCK_ORDERS.update(build_mock_orders())
