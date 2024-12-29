import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# 参数设置
data_folder = "日级行情"  # 数据文件夹，存放日级行情数据
stock_list_file = "股票列表及其所属行业.xlsx"  # 股票列表文件，包含股票代码、名称及行业
result_file = "回测结果_2024.xlsx"  # 回测结果输出文件
cache_file = "processed_data.pkl"  # 缓存文件路径
initial_cash = 100000  # 初始资金设定为 100,000
start_year = 2024  # 设置回测的开始年份为 2024


# 加载股票列表函数
def load_stock_list(file_path):
    try:
        stock_list = pd.read_excel(file_path)
        required_columns = {'ts_code', 'name', 'industry'}
        if not required_columns.issubset(stock_list.columns):
            raise ValueError(f"股票列表文件缺少必要的列：{required_columns}")
        return stock_list
    except Exception as e:
        print(f"加载股票列表文件失败：{e}")
        return pd.DataFrame()


# 加载行情数据并保存缓存函数
def load_data_optimized(folder_path, year, cache_file):
    if os.path.exists(cache_file):  # 如果缓存文件存在
        print(f"加载缓存文件：{cache_file}")
        return pd.read_pickle(cache_file)  # 加载缓存文件

    print("缓存文件不存在，开始加载原始数据...")
    data = []  # 用于存储所有股票数据
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith(('.xlsx', '.xls')):
            file_path = os.path.join(folder_path, file)
            df = pd.read_excel(file_path)
            if {'ts_code', 'trade_date', 'open', 'close', 'pre_close', 'amount'}.issubset(df.columns):
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                df = df[df['trade_date'].dt.year == year]  # 筛选指定年份数据
                if not df.empty:
                    data.append(df)
    if not data:
        print("未找到有效数据。")
        return pd.DataFrame()

    data = pd.concat(data, ignore_index=True)  # 合并所有数据
    data.to_pickle(cache_file)  # 保存为缓存文件
    print(f"数据加载完成并保存至缓存文件：{cache_file}")
    return data


# 策略回测函数
# 策略回测函数
def backtest(data):
    data.sort_values(['trade_date', 'ts_code'], inplace=True)
    data['amount_change'] = data.groupby('ts_code')['amount'].pct_change()
    data['ma5'] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(window=5).mean())
    data['pct_chg'] = data.groupby('ts_code')['close'].pct_change() * 100

    cash = initial_cash
    position = 0
    holding = None
    buy_price = 0
    results = []
    prev_day_assets = initial_cash  # 用于计算持股收益

    unique_dates = data['trade_date'].unique()

    for current_date in unique_dates[1:]:
        daily_result = {
            '日期': current_date, '现金余额': cash, '持仓股票代码': None,
            '持仓股票名称': None, '行业': None, '买入/卖出价格': None,
            '买入/卖出方向': None, '持仓数量': position, '持仓市值': 0,
            '当日总资产': cash, '交易利润': 0, '持股收益': 0,  # 添加持股收益
            '当天开盘价': None, '当天收盘价': None, '当天五日均线': None, '前一天收盘价': None
        }

        can_trade = True  # 设置标志，表示今天是否可以交易

        # 卖出逻辑
        if holding and can_trade:
            holding_data = data[(data['trade_date'] == current_date) & (data['ts_code'] == holding)]
            if not holding_data.empty:
                if holding_data.iloc[0]['close'] < holding_data.iloc[0]['open']:
                    sell_price = (holding_data.iloc[0]['open'] + holding_data.iloc[0]['close']) / 2
                    profit = position * (sell_price - buy_price)
                    cash += position * sell_price
                    position = 0
                    daily_result.update({
                        '持仓股票代码': holding,
                        '买入/卖出价格': sell_price,
                        '买入/卖出方向': '卖出',
                        '交易利润': profit,
                        '当天开盘价': holding_data.iloc[0]['open'],
                        '当天收盘价': holding_data.iloc[0]['close'],
                        '当天五日均线': holding_data.iloc[0]['ma5'],
                        '前一天收盘价': holding_data.iloc[0]['pre_close'],
                        '现金余额': cash
                    })
                    holding = None
                    can_trade = False  # 卖出后，设置为 False，不能再进行买入操作

        # 买入逻辑
        if not holding and can_trade:
            potential_stocks = data[(data['trade_date'] == current_date) & (data['amount_change'] > 4)]
            for ts_code, group in potential_stocks.groupby('ts_code'):
                current_day_data = data[(data['ts_code'] == ts_code) & (data['trade_date'] == current_date)]
                if not current_day_data.empty:
                    current_day_open = current_day_data.iloc[0]['open']
                    current_day_close = current_day_data.iloc[0]['close']
                    current_day_ma5 = current_day_data.iloc[0]['ma5']

                    if current_day_open > current_day_ma5 and current_day_close > current_day_ma5:
                        buy_price = current_day_open
                        max_hands = int(cash // (buy_price * 100))
                        if max_hands > 0:
                            position = max_hands * 100
                            cash -= position * buy_price
                            daily_result.update({
                                '持仓股票代码': ts_code,
                                '买入/卖出价格': buy_price,
                                '买入/卖出方向': '买入',
                                '持仓数量': position,
                                '当天开盘价': current_day_open,
                                '当天收盘价': current_day_close,
                                '当天五日均线': current_day_ma5,
                                '现金余额': cash
                            })
                            holding = ts_code
                            can_trade = False  # 买入后，设置为 False，不能再进行卖出操作
                            break

        if holding:
            holding_data = data[(data['trade_date'] == current_date) & (data['ts_code'] == holding)]
            if not holding_data.empty:
                holding_value = position * holding_data.iloc[0]['close']
                daily_result['持仓市值'] = holding_value
                daily_result['当日总资产'] = cash + holding_value

                # 计算持股收益：当天市值减去上一个交易日的市值
                holding_profit = holding_value - prev_day_assets
                daily_result['持股收益'] = holding_profit

                prev_day_assets = holding_value  # 更新前一天的持仓市值

        else:
            daily_result['当日总资产'] = cash

        # 每一天结束后重置 can_trade 为 True 以便于第二天可以继续交易
        results.append(daily_result)
        can_trade = True  # 重置标志为 True，表示明天可以进行交易

    return results


# 绘制盈利百分比曲线
def plot_profit_curve(results):
    equity_df = pd.DataFrame(results)
    equity_df['盈利百分比'] = ((equity_df['当日总资产'] - initial_cash) / initial_cash) * 100
    plt.figure(figsize=(10, 6))
    plt.plot(equity_df['日期'], equity_df['盈利百分比'], label="盈利百分比", color="green")
    plt.title("盈利百分比曲线")
    plt.xlabel("日期")
    plt.ylabel("盈利百分比 (%)")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 主程序调用
def main():
    stock_list = load_stock_list(stock_list_file)
    if stock_list.empty:
        print("未加载股票列表，程序终止。")
        return

    data = load_data_optimized(data_folder, start_year, cache_file)
    if data.empty:
        print("未找到有效的行情数据。")
        return

    results = backtest(data)
    results_df = pd.DataFrame(results)
    # 合并股票名称和行业信息
    results_df = results_df.merge(stock_list, left_on='持仓股票代码', right_on='ts_code', how='left')
    results_df.drop(columns=['ts_code'], inplace=True)
    results_df.rename(columns={'name': '股票名称', 'industry': '所属行业'}, inplace=True)

    results_df.to_excel(result_file, index=False)
    print(f"回测完成，结果保存至 {result_file}")

    plot_profit_curve(results)


if __name__ == "__main__":
    main()
