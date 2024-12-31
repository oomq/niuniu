# 导入所需的库
import pandas as pd  # 用于处理数据
import os  # 用于文件操作
import matplotlib.pyplot as plt  # 用于绘图
from tqdm import tqdm  # 用于显示进度条
from multiprocessing import Pool  # 用于多进程并行处理
import numpy as np  # 用于数值计算

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
        stock_list = pd.read_excel(file_path)  # 读取股票列表的Excel文件
        required_columns = {'ts_code', 'name', 'industry'}  # 必须包含的列
        if not required_columns.issubset(stock_list.columns):  # 如果缺少必要的列，抛出错误
            raise ValueError(f"股票列表文件缺少必要的列：{required_columns}")
        return stock_list  # 返回股票列表
    except Exception as e:
        print(f"加载股票列表文件失败：{e}")  # 如果加载过程中出错，打印错误信息
        return pd.DataFrame()  # 返回空的DataFrame

# 多进程加载行情数据并保存缓存函数
def process_file(file_path, year):
    try:
        df = pd.read_excel(file_path)  # 读取数据文件
        if {'ts_code', 'trade_date', 'open', 'close', 'pre_close', 'amount'}.issubset(df.columns):  # 检查文件是否包含必要的列
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')  # 将日期列转换为日期格式
            df = df[df['trade_date'].dt.year == year]  # 筛选出年份为指定年份的数据
            return df if not df.empty else None  # 返回数据，如果为空则返回None
    except Exception as e:
        print(f"处理文件 {file_path} 出错：{e}")  # 处理文件时出现错误，打印错误信息
    return None  # 如果出现错误，返回None

# 多进程加载数据并保存缓存
def load_data_with_multiprocessing(folder_path, year, cache_file):
    if os.path.exists(cache_file):  # 如果缓存文件已存在
        print(f"加载缓存文件：{cache_file}")  # 输出缓存文件加载信息
        return pd.read_pickle(cache_file)  # 加载缓存文件并返回数据

    print("缓存文件不存在，开始加载原始数据...")  # 如果缓存文件不存在，开始加载原始数据
    files = [
        os.path.join(folder_path, file)  # 获取文件夹下所有文件路径
        for file in os.listdir(folder_path)  # 遍历文件夹
        if file.endswith(('.xlsx', '.xls'))  # 只选择Excel文件
    ]

    with Pool() as pool:  # 使用多进程池处理文件
        results = list(tqdm(pool.imap(lambda f: process_file(f, year), files), total=len(files)))  # 并行处理每个文件

    data = pd.concat(filter(None, results), ignore_index=True) if results else pd.DataFrame()  # 合并非空的处理结果
    if not data.empty:
        data.to_pickle(cache_file)  # 将处理后的数据保存为缓存文件
        print(f"数据加载完成并保存至缓存文件：{cache_file}")
    else:
        print("未找到有效数据。")  # 如果没有有效数据，输出提示
    return data  # 返回加载后的数据

# 策略回测函数
def backtest(data):
    data.sort_values(['trade_date', 'ts_code'], inplace=True)
    data['amount_change'] = data.groupby('ts_code')['amount'].pct_change()  # 计算成交量变化率
    data['ma5'] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(window=5).mean())
    data['pct_chg'] = data.groupby('ts_code')['close'].pct_change() * 100

    cash = initial_cash
    position = 0
    holding = None
    buy_price = 0
    results = []
    prev_day_assets = initial_cash  # 用于计算持股收益
    max_equity = initial_cash  # 用于计算最大回撤
    drawdowns = []  # 存储每次回撤

    filter_code = []

    trade_count = 0  # 总交易次数
    win_trades = 0  # 盈利交易次数
    profit_list = []  # 记录每笔交易盈亏
    cumulative_returns = []  # 累计收益率记录

    unique_dates = data['trade_date'].unique()

    for i in range(1, len(unique_dates)):  # 从第二个日期开始
        current_date = unique_dates[i]
        previous_date = unique_dates[i - 1]  # 前一交易日

        daily_result = {
            '日期': current_date, '现金余额': cash, '持仓股票代码': None,
            '买入/卖出价格': None, '买入/卖出方向': None, '持仓数量': position,
            '持仓市值': 0, '当日总资产': cash, '交易利润': None, '持股收益': None,
            '开盘价': None, '收盘价': None, '五日均线': None, '昨收价': None,
            '符合条件的股票代码':filter_code
        }

        can_trade = True

        # 卖出逻辑
        if holding and can_trade:
            holding_data = data[(data['trade_date'] == current_date) & (data['ts_code'] == holding)]
            previous_day_holding_data = data[(data['ts_code'] == holding) & (data['trade_date'] == previous_date)]

            if not holding_data.empty and not previous_day_holding_data.empty:
                if holding_data.iloc[0]['close'] < holding_data.iloc[0]['open'] and holding_data.iloc[0]['close']< holding_data.iloc[0]['ma5']:
                    sell_price = holding_data.iloc[0]['close']
                    profit = position * (sell_price - buy_price)
                    profit_pct = profit / (position * buy_price) * 100
                    cash += position * sell_price
                    position = 0
                    trade_count += 1
                    profit_list.append(profit)
                    if profit > 0:
                        win_trades += 1
                    daily_result.update({
                        '持仓股票代码': holding,
                        '买入/卖出价格': sell_price,
                        '买入/卖出方向': '卖出',
                        '交易利润': f"{profit:.2f}（{profit_pct:.2f}%）",
                        '开盘价': holding_data.iloc[0]['open'],
                        '收盘价': holding_data.iloc[0]['close'],
                        '五日均线': holding_data.iloc[0]['ma5'],
                        '昨收价': holding_data.iloc[0]['pre_close'],
                        '现金余额': cash
                    })
                    holding = None
                    can_trade = False
            else:
                print("selling-error")


        # 买入逻辑 - 基于前一日的成交量变化率
        if not holding and can_trade:
            # 获取前一日的成交量变化率大于 400% 的股票
            potential_stocks = data[(data['trade_date'] == previous_date) & (data['amount_change'] > 4)]

            for ts_code, group in potential_stocks.groupby('ts_code'):
                current_day_data = data[(data['ts_code'] == ts_code) & (data['trade_date'] == current_date)]
                previous_day_data = data[(data['ts_code'] == ts_code) & (data['trade_date'] == previous_date)]
                if not current_day_data.empty:
                    current_day_open = current_day_data.iloc[0]['open']
                    current_day_close = current_day_data.iloc[0]['close']
                    current_day_ma5 = current_day_data.iloc[0]['ma5']
                    previous_day_open = previous_day_data.iloc[0]['open']
                    previous_day_close = previous_day_data.iloc[0]['close']
                    previous_day_ma5=previous_day_data.iloc[0]['ma5']

                    if (current_day_open > previous_day_ma5 and previous_day_close > previous_day_ma5
                            and previous_day_close>previous_day_open):###todo：改为连续前几天收盘价大于五日线
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
                                '开盘价': current_day_open,
                                '收盘价': current_day_close,
                                '五日均线': current_day_ma5,
                                '现金余额': cash
                            })
                            holding = ts_code
                            can_trade = False
                            break

        if holding:
            holding_data = data[(data['trade_date'] == current_date) & (data['ts_code'] == holding)]
            if not holding_data.empty:
                holding_value = position * holding_data.iloc[0]['close']
                holding_profit = holding_value - prev_day_assets
                holding_profit_pct = (holding_profit / prev_day_assets) * 100

                daily_result.update({
                    '持仓市值': holding_value,
                    '当日总资产': cash + holding_value,
                    '持股收益': f"{holding_profit:.2f}（{holding_profit_pct:.2f}%）",
                    '开盘价': holding_data.iloc[0]['open'],
                    '收盘价': holding_data.iloc[0]['close'],
                    '五日均线': holding_data.iloc[0]['ma5'],
                    '昨收价': holding_data.iloc[0]['pre_close'],
                })
                prev_day_assets = holding_value
            else:  # 假设当天停牌，数据沿用上一交易日
                daily_result.update({
                    '持仓市值': holding_value,
                    '当日总资产': cash + holding_value,
                    '持股收益': f"0（0%）",
                    '开盘价': holding_data.iloc[0]['open'],
                    '收盘价': holding_data.iloc[0]['close'],
                    '五日均线': holding_data.iloc[0]['ma5'],
                    '昨收价': holding_data.iloc[0]['pre_close'],
                })

        else:
            daily_result['当日总资产'] = cash

        # 计算最大回撤
        current_equity = daily_result['当日总资产']
        max_equity = max(max_equity, current_equity)
        drawdown = (max_equity - current_equity) / max_equity
        drawdowns.append(drawdown)

        # 记录累计收益
        cumulative_return = (current_equity - initial_cash) / initial_cash * 100
        cumulative_returns.append({'日期': current_date, '累计收益率': cumulative_return})

        results.append(daily_result)
        can_trade = True

    # 回测统计
    max_drawdown = max(drawdowns) * 100 if drawdowns else 0
    win_rate = win_trades / trade_count * 100 if trade_count > 0 else 0
    profit_factor = (sum(p for p in profit_list if p > 0) /
                     abs(sum(p for p in profit_list if p < 0))) if any(p < 0 for p in profit_list) else float('inf')

    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"总交易次数: {trade_count}")
    print(f"交易胜率: {win_rate:.2f}%")
    print(f"盈亏比: {profit_factor:.2f}")

    # 绘制盈利曲线
    plot_cumulative_returns(cumulative_returns)

    return results

# 绘制累计收益率曲线
def plot_cumulative_returns(cumulative_returns):
    # 将收益数据转换为 DataFrame
    df = pd.DataFrame(cumulative_returns)
    df['日期'] = pd.to_datetime(df['日期'])
    df.sort_values('日期', inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df['日期'], df['累计收益率'], label='累计收益率 (%)', color='blue')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('累计收益率曲线')
    plt.xlabel('日期')
    plt.ylabel('累计收益率 (%)')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.show()

# 主流程
def main():
    # 加载股票列表
    stock_list = load_stock_list(stock_list_file)
    if stock_list.empty:
        return

    # 加载行情数据
    data = load_data_with_multiprocessing(data_folder, start_year, cache_file)
    if data.empty:
        print("无有效数据，终止回测流程。")
        return

    # 执行回测
    results = backtest(data)

    # 保存结果至 Excel，合并股票信息
    results_df = pd.DataFrame(results)
    results_df = results_df.merge(stock_list, left_on='持仓股票代码', right_on='ts_code', how='left')
    results_df.drop(columns=['ts_code','symbol','list_date'], inplace=True)
    results_df.rename(columns={'name': '股票名称', 'industry': '所属行业'}, inplace=True)

    results_df.to_excel(result_file, index=False)  # 保存回测结果到Excel
    print(f"回测结果已保存至 {result_file}")

# 启动主流程
if __name__ == "__main__":
    main()