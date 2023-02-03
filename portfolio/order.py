from pydantic import BaseModel, StrictStr, validator
from typing import Optional, Union, List, Any
from decimal import Decimal


class OrderMethod(BaseModel):
    order_type: str
    price_type: str
    stop_followed_price: str
    delay_point: float = 0
    delay_type: str = "down"
    stop_loss: float = 0
    take_profit: float = 0

    @validator("order_type")
    def order_type_rules(cls, v, values, **kwargs):
        if v not in ["normal", "touch"]:
            raise ValueError("order_type should be one of ['normal', 'touch']")
        return v

    @validator("price_type")
    def price_type_rules(cls, v, values, **kwargs):
        if v not in ["market", "limit"]:
            raise ValueError("price_type should be one of ['market', 'limit']")
        return v

    @validator("stop_followed_price")
    def stop_followed_price_rules(cls, v, values, **kwargs):
        stop_followed_price = ["entry_price", "exit_price", "fall_max", "rise_max"]
        if v not in stop_followed_price:
            raise ValueError(
                f"stop_followed_price should be one of {stop_followed_price}"
            )
        return v


class OrderPlan(BaseModel):
    entry_order: OrderMethod
    sp_exit_order: Optional[OrderMethod]
    tp_exit_order: Optional[OrderMethod]
    exit_order: OrderMethod


class Order:
    def __init__(self, signal: dict) -> None:
        self.tags = self.gen_tags_from_signal(signal)
        self.signal = signal

    def gen_tags_from_signal(self, signal: dict) -> List[dict]:
        temp = []
        for i, order_plan in enumerate(signal["order_plans"]):
            for order_kind, enex in order_plan.items():
                if enex:
                    infos = dict(
                        order_kind=order_kind,
                        strategy_name=signal["name"],
                        entry_time=signal["entry_time"],
                        symbol=signal["symbol"],
                        product=signal["product"],
                    )
                    enex.update(infos)
                    enex["group"] = str(i)
                    if enex not in temp:
                        temp.append(enex)
        return temp

    def gen_orders(self) -> dict:
        data = self.signal
        orders = {}
        for tag in self.tags:
            group = tag["group"]
            orders[group] = orders.get(group, {})
            delay_point = tag["delay_point"]
            delay_type = tag["delay_type"]
            stop_loss = tag["stop_loss"]
            take_profit = tag["take_profit"]
            order_kind = tag["order_kind"]
            order_type = tag["order_type"]
            stop_followed_price = tag["stop_followed_price"]
            price_type = tag["price_type"]
            cash = int(1)
            cols = [v for k, v in tag.items() if isinstance(v, str)]
            tag = "|".join(cols)
            if "entry_order" == order_kind:
                delay_adjust = 1 if delay_type == "up" else -1
                delay_point = Decimal(delay_point) * delay_adjust
                order = dict(
                    order_kind=order_kind,
                    symbol=data["symbol"],
                    product=data["product"],
                    is_stop_order=True if "touch" in order_type else False,
                    stop_followed_price=stop_followed_price,
                    stop_price=Decimal(data[stop_followed_price])
                    + Decimal(delay_point),
                    followed_price=Decimal(data[stop_followed_price]),
                    cash=cash,
                    direction="Buy" if data["direction"] in ["L", "l"] else "Sell",
                    price_type=price_type,
                )
                order["tag"] = tag
                orders[group][order_kind] = order
            elif "sp_exit_order" == order_kind:
                sp_adjust = 1 if data["direction"] not in ["L", "l"] else -1
                sp_point = Decimal(stop_loss) * sp_adjust
                order = dict(
                    order_kind=order_kind,
                    symbol=data["symbol"],
                    product=data["product"],
                    is_stop_order=True if "touch" in order_type else False,
                    stop_followed_price=stop_followed_price,
                    stop_price=Decimal(data[stop_followed_price]) + sp_point,
                    followed_price=Decimal(data[stop_followed_price]),
                    cash=cash,
                    direction="Buy" if data["direction"] not in ["L", "l"] else "Sell",
                    price_type=price_type,
                )
                order["tag"] = tag
                orders[group][order_kind] = order
            elif "tp_exit_order" == order_kind:
                tp_adjust = 1 if data["direction"] in ["L", "l"] else -1
                tp_point = Decimal(take_profit) * tp_adjust
                order = dict(
                    order_kind=order_kind,
                    symbol=data["symbol"],
                    product=data["product"],
                    is_stop_order=True if "touch" in order_type else False,
                    stop_followed_price=stop_followed_price,
                    stop_price=Decimal(data[stop_followed_price]) + tp_point,
                    followed_price=Decimal(data[stop_followed_price]),
                    cash=cash,
                    direction="Buy" if data["direction"] not in ["L", "l"] else "Sell",
                    price_type=price_type,
                )
                order["tag"] = tag
                orders[group][order_kind] = order
            elif "exit_order" == order_kind:
                order = dict(
                    order_kind=order_kind,
                    symbol=data["symbol"],
                    product=data["product"],
                    is_stop_order=True if "touch" in order_type else False,
                    stop_followed_price=stop_followed_price,
                    stop_price=Decimal(data[stop_followed_price]),
                    followed_price=Decimal(data[stop_followed_price]),
                    cash=cash,
                    direction="Buy" if data["direction"] not in ["L", "l"] else "Sell",
                    price_type=price_type,
                )
                order["tag"] = tag
                orders[group][order_kind] = order
        return orders
