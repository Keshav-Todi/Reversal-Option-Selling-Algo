#pip install delta_rest_client
from delta_rest_client import DeltaRestClient
import pandas as pd
import time
import requests
from datetime import datetime, timedelta
import json
import csv
import requests
from requests.exceptions import RequestException
import numpy as np

log_file_path_csv_1 = 'C:/Users/ktcan/Downloads/crypto files/option_chain_storing_'
#log_file_path_csv_1 = '/home/ubuntu/Desktop/algoedge-main-updated-latest/order_book/option_chain_storing_'
log_file_path_csv_2=''

def write_to_csv_file(data_list, file_suffix):
    """
    Writes a list of dictionaries (data_list) to a CSV file with headers.
    """
    global log_file_path_csv_1
    #global log_file_path_csv_2

    log_file_path_csv_2 = log_file_path_csv_1 + file_suffix + '_.csv'

    # Extract headers from the keys of the first dictionary
    headers = data_list[0].keys() if data_list else []

    with open(log_file_path_csv_2, 'a', newline='') as log_file:
        writer = csv.DictWriter(log_file, fieldnames=headers)
        
        # Write header only if the file is empty
        log_file.seek(0, 2)  # Move to the end of the file to check if it's empty
        if log_file.tell() == 0:
            writer.writeheader()

        # Write each dictionary in the list as a new row
        for data in data_list:
            writer.writerow(data)

# Example call to write to CSV
#write_to_csv_file(data_list3, 'example_suffix')  # Provide a suffix for the file name

def read_csv_with_headers(file_suffix):
    file_path = log_file_path_csv_1 + file_suffix + '_.csv'
    """
    Reads a CSV file and returns a list of dictionaries, where each dictionary represents a row.
    """
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = [row for row in csv_reader]
    return data

# Example call to read from CSV
#file_path = 'C:/Users/ktcan/Downloads/crypto files/option_chain_storing_example_suffix_.csv'
#data_list = read_csv_with_headers(file_path)
#print(data_list)

def clear_log_file_csv(file_suffix):
    file_path = log_file_path_csv_1 + file_suffix + '_.csv'
    """Clears the contents of the log file."""
    open(file_path, 'w').close()  # Opening in 'w' mode and closing the file clears it

def option_chain_internal(data_list, mark_price):

    max_loss=12

    data_list3=[]
    mark_price=float(mark_price)

    option_data_C,nearest_index,is_extreme=get_nearest_strike_price(data_list,mark_price,'C')
    atm_bid_c=float(option_data_C['bid'])
    atm_ask_c=float(option_data_C['ask'])
    atm_strike_c=float(option_data_C['strike'])

    option_data_P,nearest_index,is_extreme=get_nearest_strike_price(data_list,mark_price,'P')
    atm_bid_p=float(option_data_P['bid'])
    atm_ask_p=float(option_data_P['ask'])
    atm_strike_p=float(option_data_P['strike'])

    for option_data in data_list:
        # Depending on whether the option type is 'C' (Call) or 'P' (Put), choose the appropriate fields
        data_list3.append({
                    'index': option_data['index'],
                    'call_symbol': option_data['call_symbol'],
                    'call_id': option_data['call_id'],
                    'rr_bid_c':atm_bid_c/float(option_data['call_bid']),
                    'dis_bid_c':(float(option_data['strike'])/atm_strike_c-1)*100,
                    'call_bid': option_data['call_bid'],
                    'call_ask': option_data['call_ask'],
                    'strike': option_data['strike'],
                    'put_symbol': option_data['put_symbol'],
                    'put_id': option_data['put_id'],
                    'rr_bid_p':atm_bid_p/float(option_data['put_bid']),
                    'dis_bid_p':(1-float(option_data['strike'])/atm_strike_c)*100,
                    'put_bid': option_data['put_bid'],
                    'put_ask': option_data['put_ask']
                })
        #print(f"{option_data['index']} - [{option_data['call_symbol']} ID: {option_data['call_id']} RR {(atm_bid_c/float(option_data['call_bid'])):.2f} Dis: {((float(option_data['strike'])/atm_strike_c-1)*100):.2f} Bid: {option_data['call_bid']} Ask: {option_data['call_ask']}] {option_data['strike']} [{option_data['put_symbol']} ID: {option_data['put_id']} RR {(atm_bid_p/float(option_data['put_bid'])):.2f} Dis: {((1-float(option_data['strike'])/atm_strike_c)*100):.2f} Bid: {option_data['put_bid']} Ask: {option_data['put_ask']}]")
        print(f"{option_data['index']} - [{option_data['call_symbol']} ID: {option_data['call_id']} RR {(atm_bid_c/float(option_data['call_bid'])):.2f} Dis: {((float(option_data['strike'])/atm_strike_c-1)*100):.2f} Qt: {(max_loss/(atm_bid_c-float(option_data['call_bid'])+1)):.3f} Bid: {option_data['call_bid']} Ask: {option_data['call_ask']}] {option_data['strike']} [{option_data['put_symbol']} ID: {option_data['put_id']} RR {(atm_bid_p/float(option_data['put_bid'])):.2f} Dis: {((1-float(option_data['strike'])/atm_strike_c)*100):.2f} Qt: {(max_loss/(atm_bid_p-float(option_data['put_bid'])+1)):.3f} Bid: {option_data['put_bid']} Ask: {option_data['put_ask']}]")

    
    #print(data_list3)

    return data_list3

def option_chain_internal_1(data_list, mark_price,distance_1,distance_2):

    option_data_C,atm_index,is_extreme=get_nearest_strike_price(data_list,float(mark_price),'C')
    max_loss=12

    data_list3=[]
    mark_price=float(mark_price)
    
    distance_step=750
    distance=distance_1
    cntt=0
    tot_outer=0
    tot_inner=len(data_list)
    
    while distance<=distance_2:
        distance=distance+distance_step
        tot_outer=tot_outer+1
    
    tot_outer=tot_outer*2
    
    result_array = np.zeros((tot_inner, tot_outer))

    distance=distance_1


    while distance<=distance_2:

        inner_array_1 = []
        inner_array_2 = []

        for option_data in data_list:

            strike=option_data['strike']
            dis_c=float(strike)-distance
            dis_p=float(strike)-distance

            option_data_C,nearest_index,is_extreme=get_nearest_strike_price(data_list,dis_c,'C')
            bid_c=float(option_data_C['bid'])
            ask_c=float(option_data_C['ask'])
            strike_c=float(option_data_C['strike'])

            option_data_P,nearest_index,is_extreme=get_nearest_strike_price(data_list,dis_p,'P')
            bid_p=float(option_data_P['bid'])
            ask_p=float(option_data_P['ask'])
            strike_p=float(option_data_P['strike'])


            # Depending on whether the option type is 'C' (Call) or 'P' (Put), choose the appropriate fields
            data_list3.append({
                        'index': option_data['index'],
                        'call_symbol': option_data['call_symbol'],
                        'call_id': option_data['call_id'],
                        'rr_bid_c':bid_c/float(option_data['call_bid']),
                        'dis_bid_c':(float(option_data['strike'])/strike_c-1)*100,
                        'call_bid': option_data['call_bid'],
                        'call_ask': option_data['call_ask'],
                        'strike': option_data['strike'],
                        'put_symbol': option_data['put_symbol'],
                        'put_id': option_data['put_id'],
                        'rr_bid_p':bid_p/float(option_data['put_bid']),
                        'dis_bid_p':(1-float(option_data['strike'])/strike_c)*100,
                        'put_bid': option_data['put_bid'],
                        'put_ask': option_data['put_ask']
                    })
            #print(f"{option_data['index']} - [{option_data['call_symbol']} ID: {option_data['call_id']} RR {(atm_bid_c/float(option_data['call_bid'])):.2f} Dis: {((float(option_data['strike'])/atm_strike_c-1)*100):.2f} Bid: {option_data['call_bid']} Ask: {option_data['call_ask']}] {option_data['strike']} [{option_data['put_symbol']} ID: {option_data['put_id']} RR {(atm_bid_p/float(option_data['put_bid'])):.2f} Dis: {((1-float(option_data['strike'])/atm_strike_c)*100):.2f} Bid: {option_data['put_bid']} Ask: {option_data['put_ask']}]")
            if(bid_c>float(option_data['call_bid'])):
                print(f"{option_data['index']} - [{option_data['call_symbol']} ID: {option_data['call_id']} RR {(bid_c/float(option_data['call_bid'])):.2f} Dis: {((float(option_data['strike'])/strike_c-1)*100):.2f} Qt: {(max_loss/(bid_c-float(option_data['call_bid'])*1.001)):.3f} Bid: {bid_c} Ask: {option_data['call_bid']}] {option_data['strike']} [{option_data['put_symbol']} ID: {option_data['put_id']} RR {(-float(option_data['put_bid'])/bid_p):.2f} Dis: {((1-float(option_data['strike'])/strike_c)*100):.2f} Qt: {(max_loss/(bid_p-float(option_data['put_bid'])*1.001)):.3f} Bid: {bid_p} Ask: {option_data['put_bid']}]")
                inner_array_1.append(bid_c/float(option_data['call_bid']))
                inner_array_2.append(-float(option_data['put_bid'])/bid_p)
            else:
                print(f"{option_data['index']} - [{option_data['call_symbol']} ID: {option_data['call_id']} RR {(-float(option_data['call_bid'])/bid_c):.2f} Dis: {((float(option_data['strike'])/strike_c-1)*100):.2f} Qt: {(max_loss/(bid_c-float(option_data['call_bid'])*1.001)):.3f} Bid: {bid_c} Ask: {option_data['call_bid']}] {option_data['strike']} [{option_data['put_symbol']} ID: {option_data['put_id']} RR {(bid_p/float(option_data['put_bid'])):.2f} Dis: {((1-float(option_data['strike'])/strike_c)*100):.2f} Qt: {(max_loss/(bid_p-float(option_data['put_bid'])*1.001)):.3f} Bid: {bid_p} Ask: {option_data['put_bid']}]")
                inner_array_1.append(-float(option_data['call_bid'])/bid_c)
                inner_array_2.append(bid_p/float(option_data['put_bid']))
         
        result_array[:, cntt] = inner_array_1
        result_array[:, cntt+1] = inner_array_2

        distance=distance+distance_step
        cntt=cntt+2
    
    #print(data_list3)
    #print(result_array)

    index=0
    for row in result_array:
        if(index==atm_index):
            print('------------------------------------------------------------------------------------------------------------------')
        index=index+1
        print(" | ".join(f"{val:.2f}" for val in row))

    return data_list3


def get_nearest_option_price(data_list, price,option_type):
    best_option = None
    best_bid = float(price)  # Initialize with a low bid to compare

    for option_data in data_list:
        # Depending on whether the option type is 'C' (Call) or 'P' (Put), choose the appropriate fields
        if option_type == 'C':
            if float(option_data['call_bid']) > float(best_bid):
                best_bid = option_data['call_bid']
                best_option = {
                    'symbol': option_data['call_symbol'],
                    'id': option_data['call_id'],
                    'bid': option_data['call_bid'],
                    'ask': option_data['call_ask'],
                    'strike': option_data['strike']
                }
        elif option_type == 'P':
            if float(option_data['put_bid']) > float(best_bid):
                best_bid = option_data['put_bid']
                best_option = {
                    'symbol': option_data['put_symbol'],
                    'id': option_data['put_id'],
                    'bid': option_data['put_bid'],
                    'ask': option_data['put_ask'],
                    'strike': option_data['strike']
                }

    return best_option

def get_nearest_strike_price(data_list, price, option_type):
    nearest_option = None
    nearest_index = -1  # Initialize with an invalid index
    min_diff = float('inf')  # Initialize with a large value
    
    for index, option_data in enumerate(data_list):
        strike_price = float(option_data['strike'])
        diff = abs(strike_price - price)
        
        # Find the nearest strike price
        if diff < min_diff:
            min_diff = diff
            nearest_index = index  # Store the index of the nearest strike
            if option_type == "C":  # Call option
                nearest_option = {
                    'symbol': option_data['call_symbol'],
                    'id': option_data['call_id'],
                    'bid': option_data['call_bid'],
                    'ask': option_data['call_ask'],
                    'strike': option_data['strike']
                }
            elif option_type == "P":  # Put option
                nearest_option = {
                    'symbol': option_data['put_symbol'],
                    'id': option_data['put_id'],
                    'bid': option_data['put_bid'],
                    'ask': option_data['put_ask'],
                    'strike': option_data['strike']
                }
    
    # Check if the nearest option is at the first or last of the data_list
    is_extreme = nearest_index == 0 or nearest_index == len(data_list) - 1

    return nearest_option, nearest_index, is_extreme  # Return the option data, index, and whether it's at an extreme

def opt_chain_expiry(target_expiry_index):
        # Initialize data_list to store options information for the specific expiry
    data_list = []

    headers = {'Accept': 'application/json'}
    params = {
        'contract_types': 'call_options,put_options',
        'states': 'live'
    }

    # Example: Assume you want the options for the 1st expiry (index 0 is the first one)
    #target_expiry_index = 0  # Change this to 0, 1, 2, etc., to choose different expiries

    # Assume current spot price (this should ideally be fetched dynamically)
    current_spot_price = float(mark_price_BTCUSDT)  # Replace this with the actual mark price
    lenn = 30

    # Fetch the products from the API
    response = requests.get('https://api.delta.exchange/v2/products', params=params, headers=headers)
    products = response.json()

    # Organizing options by expiry and then by strike
    options_by_expiry = {}

    for product in products.get('result', []):
        if product['underlying_asset']['symbol'] == 'BTC':
            expiry_date = product['settlement_time']
            strike_price = float(product['strike_price'])
            option_type = 'Call' if 'call' in product['contract_type'] else 'Put'

            if expiry_date not in options_by_expiry:
                options_by_expiry[expiry_date] = {}
            if strike_price not in options_by_expiry[expiry_date]:
                options_by_expiry[expiry_date][strike_price] = {'Call': [], 'Put': []}

            options_by_expiry[expiry_date][strike_price][option_type].append(product)

    # Sort the expiries in chronological order and pick the one based on the index
    sorted_expiries = sorted(options_by_expiry.keys())
    if target_expiry_index < len(sorted_expiries):
        # Get the expiry at the target index
        target_expiry = sorted_expiries[target_expiry_index]
        print(f"Processing options for expiry: {target_expiry}")
        
        strikes = options_by_expiry[target_expiry]
        sorted_strikes = sorted(strikes.keys())
        atm_index = min(range(len(sorted_strikes)), key=lambda i: abs(sorted_strikes[i] - current_spot_price))
        start_index = max(0, atm_index - lenn)
        end_index = min(len(sorted_strikes), atm_index + (lenn + 1))

        for i, strike in enumerate(sorted_strikes[start_index:end_index], start=start_index - atm_index - lenn):
            a1 = 0
            b1 = 0
            for option_type in ['Call', 'Put']:
                for option in strikes[strike][option_type]:
                    try:
                        ticker_info = delta_client.get_ticker(option['symbol'])
                    except RequestException as e:
                        print(f"A requests-related error occurred: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                    side = "Call" if option_type == "Call" else "Put"

                    if ticker_info is not None:
                        if option_type == "Call":
                            opt_C = option
                            product_id_C=ticker_info['product_id']
                            best_bid_C = ticker_info['quotes']['best_bid']
                            best_ask_C = ticker_info['quotes']['best_ask']
                            a1 = 1
                        if option_type == "Put":
                            opt_P = option
                            product_id_P=ticker_info['product_id']
                            best_bid_P = ticker_info['quotes']['best_bid']
                            best_ask_P = ticker_info['quotes']['best_ask']
                            b1 = 1

            b = i + lenn

            if a1 == 1 and b1 == 1:
                # Store the relevant data for both Call and Put options
                data_list.append({
                    'index': b,
                    'call_symbol': opt_C['symbol'],
                    'call_id': product_id_C,
                    'call_bid': best_bid_C,
                    'call_ask': best_ask_C,
                    'strike': strike,
                    'put_symbol': opt_P['symbol'],
                    'put_id': product_id_P,
                    'put_bid': best_bid_P,
                    'put_ask': best_ask_P
                })
                print(f"{b} - [{opt_C['symbol']} ID: {opt_C['id']} Bid: {best_bid_C} Ask: {best_ask_C}] {strike} [{opt_P['symbol']} ID: {opt_P['id']} Bid: {best_bid_P} Ask: {best_ask_P}]")
    else:
        print(f"Invalid expiry index: {target_expiry_index}. Only {len(sorted_expiries)} expiries available.")

    #print(data_list)
    return data_list

def choose_best_option_with_strike_priority(data_list, mark_price_ini,direction, min_amount_ini, min_price, max_quantity):
    valid_options = []

    global start_min_amount

    mark_price=mark_price_ini
    min_amount=min_amount_ini
    min_price=float(mark_price)*0.1/100.0
        # Dynamically decide the step size if it's not provided

    if max_quantity <= 50:
        step_size = 1
    elif max_quantity <= 200:
        step_size = 5
    else:
        step_size = max(10, max_quantity // 20)  # A dynamic step size based on max_quantity

    mark_pr_gap=0.8

    if(direction=='up'):
        option_type='C'
    elif(direction=='down'):
        option_type='P'

    amount_reduce_per=15
    cnt=1
    condition=True
    while(condition):
        # Step 1: Filter options based on min_amount, min_price, and max_quantity

        try:
            mark_price=float(delta_client.get_ticker(symbol)['mark_price'])
        except RequestException as e:
            print(f"A requests-related error occurred: {e}")
            mark_price=mark_price_ini
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            mark_price=mark_price_ini


        for option_data in data_list:
            if(option_type=='C'):
                option_symbol = option_data['call_symbol']  # Using 'bid' for option price, adjust if needed
                option_id = option_data['call_id']  # Using 'bid' for option price, adjust if needed
            elif(option_type=='P'):
                option_symbol = option_data['put_symbol']  # Using 'bid' for option price, adjust if needed
                option_id = option_data['put_id']  # Using 'bid' for option price, adjust if needed
            try:
                ticker=delta_client.get_ticker(option_symbol)
            except RequestException as e:
                print(f"A requests-related error occurred: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            best_bid=float(ticker['quotes']['best_bid'])
            strike_price = float(option_data['strike'])

            # Check if the option price meets the minimum price requirement
            if ((strike_price > (mark_price*(100+mark_pr_gap)/100) and option_type=='C')or(strike_price < (mark_price*(100-mark_pr_gap)/100) and option_type=='P')):
                if best_bid >= min_price:
                    for quantity in range(1, int(max_quantity) + 1, int(step_size)):
                        option_amount = best_bid * quantity/1000.0  # Total amount for this option and quantity
                        
                        # Check if the option amount meets the minimum required amount
                        if option_amount >= min_amount:
                            valid_options.append({
                                'strike': strike_price,
                                'symbol': option_symbol,
                                'id':option_id,
                                'price': best_bid,
                                'quantity': quantity,
                                'actual_qt':quantity/1000.0,
                                'option_amount': option_amount
                            })

        # Step 2: Apply priority based on farthest strike price
        if valid_options:
            condition=False
        else:
            min_amount=min_amount_ini*(100-amount_reduce_per*cnt)/100
            cnt+=1
            print(f'no valid options found, continuing with new min amount {min_amount}')
    
    start_min_amount=min_amount

    # Initialize the best option with the first valid option
    best_option = valid_options[0]

    print(f'valid options are: {len(valid_options)}')
    #print()
    #print(valid_options)
    #print()
    
    for option in valid_options:
        if option_type == 'C' and option['strike'] > best_option['strike']:
            # For Call options, choose the option with the highest strike price
            best_option = option
        elif option_type == 'P' and option['strike'] < best_option['strike']:
            # For Put options, choose the option with the lowest strike price
            best_option = option

    print('best option found is: ')
    print(best_option)

    return best_option

def choose_best_option_with_strike_priority_1(data_list, mark_price_ini,direction, max_loss, min_price, max_quantity):
    valid_options = []

    mark_price=mark_price_ini
    min_amount_ini=max_loss/2.5
    min_amount=min_amount_ini
    min_price=float(mark_price)*0.1/100.0
        # Dynamically decide the step size if it's not provided

    if max_quantity <= 50:
        step_size = 1
    elif max_quantity <= 200:
        step_size = 5
    else:
        step_size = max(10, max_quantity // 20)  # A dynamic step size based on max_quantity

    mark_pr_gap=0.8

    if(direction=='up'):
        option_type='C'
    elif(direction=='down'):
        option_type='P'


    option_data_C,nearest_index,is_extreme=get_nearest_strike_price(data_list,float(mark_price),'C')
    atm_bid_c=float(option_data_C['bid'])
    atm_ask_c=float(option_data_C['ask'])
    atm_strike_c=float(option_data_C['strike'])

    option_data_P,nearest_index,is_extreme=get_nearest_strike_price(data_list,float(mark_price),'P')
    atm_bid_p=float(option_data_P['bid'])
    atm_ask_p=float(option_data_P['ask'])
    atm_strike_p=float(option_data_P['strike'])

    amount_reduce_per=15
    cnt=1
    condition=True
    while(condition):
        # Step 1: Filter options based on min_amount, min_price, and max_quantity

        try:
            mark_price=float(delta_client.get_ticker(symbol)['mark_price'])
        except RequestException as e:
            print(f"A requests-related error occurred: {e}")
            mark_price=mark_price_ini
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            mark_price=mark_price_ini


        for option_data in data_list:
            if(option_type=='C'):
                option_symbol = option_data['call_symbol']  # Using 'bid' for option price, adjust if needed
                option_id = option_data['call_id']  # Using 'bid' for option price, adjust if needed
            elif(option_type=='P'):
                option_symbol = option_data['put_symbol']  # Using 'bid' for option price, adjust if needed
                option_id = option_data['put_id']  # Using 'bid' for option price, adjust if needed
            
            ticker=delta_client.get_ticker(option_symbol)
            best_bid=float(ticker['quotes']['best_bid'])
            strike_price = float(option_data['strike'])

            # Check if the option price meets the minimum price requirement
            if ((strike_price > (mark_price*(100+mark_pr_gap)/100) and option_type=='C')or(strike_price < (mark_price*(100-mark_pr_gap)/100) and option_type=='P')):
                if best_bid >= min_price:
                    if(option_type=='C'):
                        max_quantity=max_loss/(atm_bid_c-best_bid)
                    elif(option_type=='P'):
                        max_quantity=max_loss/(atm_bid_p-best_bid)
                    for quantity in range(1, max_quantity + 1, step_size):
                        option_amount = best_bid * quantity/1000.0  # Total amount for this option and quantity
                        
                        # Check if the option amount meets the minimum required amount
                        if option_amount >= min_amount:
                            valid_options.append({
                                'strike': strike_price,
                                'symbol': option_symbol,
                                'id':option_id,
                                'price': best_bid,
                                'quantity': quantity,
                                'actual_qt':quantity/1000.0,
                                'option_amount': option_amount
                            })

        # Step 2: Apply priority based on farthest strike price
        if valid_options:
            condition=False
        else:
            min_amount=min_amount_ini*(100-amount_reduce_per*cnt)/100
            cnt+=1
            print(f'no valid options found, continuing with new min amount {min_amount}')

    # Initialize the best option with the first valid option
    best_option = valid_options[0]

    print('valid options are:')
    print()
    print(valid_options)
    print()
    
    for option in valid_options:
        if option_type == 'C' and option['strike'] > best_option['strike']:
            # For Call options, choose the option with the highest strike price
            best_option = option
        elif option_type == 'P' and option['strike'] < best_option['strike']:
            # For Put options, choose the option with the lowest strike price
            best_option = option

    print('best option found is: ')
    print(best_option)

    return best_option

def my_function():
    print("Reset time!")

#"""
delta_client = DeltaRestClient(
  base_url='https://api.india.delta.exchange',#'https://api.delta.exchange',#'https://cdn.india.deltaex.org',#
  api_key='',
  api_secret=''
  ,raise_for_status=False
)
#"""
#clear_log_file()  # This will clear the log file
#clear_log_file_csv()  # This will clear the log file

"""
symbol = 'P-BTC-64200-210424'
ticker = delta_client.get_ticker(symbol)
mark_price = ticker['mark_price']
product_id = ticker['product_id']
quotes=ticker['quotes']
best_bid=quotes['best_bid']
best_ask=quotes['best_ask']
#products = delta_client.get_products()
"""

symbol = 'BTCUSD'
ticker = delta_client.get_ticker(symbol)
#print(ticker)
mark_price_BTCUSDT = ticker['mark_price']
product_id_BTCUSDT = ticker['product_id']
print(mark_price_BTCUSDT)
print(product_id_BTCUSDT)



ticker = delta_client.get_ticker(symbol)
current_spot_price = ticker['mark_price']


condition=True
protect_trigger=0
entry_size=0
entry_price=0
quantity_1=2

mark_price=delta_client.get_ticker(symbol)['mark_price']
intital_start_price=mark_price
price_trigger_b=float(intital_start_price)+30
price_trigger_s=float(intital_start_price)-30
price_trigger_near_b=float(intital_start_price)+1000
price_trigger_near_s=float(intital_start_price)-1000
side='sell'

near_b=1
near_s=1
new_trigger=0
needed_c=0
needed_p=0

interval = 180  # Set the interval in seconds (e.g., 15 or 30 seconds)
next_call = time.time()+interval  # Set the initial time


last_time = ((read_csv_with_headers('time_1'))[0])['time']
curr_time=time.time()

expiry_date=0

if(curr_time>=(float(last_time)+600)):
    print(f'no option found recent, curr_time {curr_time}, last time {last_time}')
    data_list =opt_chain_expiry(expiry_date)
    clear_log_file_csv('temp_1')
    clear_log_file_csv('time_1')
    next_call_d=[]
    next_call_d.append({'time':time.time()})
    write_to_csv_file(data_list, 'temp_1')  # Provide a suffix for the file name
    write_to_csv_file(next_call_d, 'time_1')  # Provide a suffix for the file name
else:
    print(f'found already stored, curr_time {curr_time}, last time {last_time}')
    data_list = read_csv_with_headers('temp_1')
    option_chain_internal(data_list,mark_price)


strike_chosen_b=float(mark_price)#price_trigger_b+1000
strike_chosen_s=float(mark_price)#price_trigger_s-1000
nearest_option_info_c, nearest_opt_index_c,is_extreme_c = get_nearest_strike_price(data_list, strike_chosen_b, 'C')
nearest_option_info_p, nearest_opt_index_p,is_extreme_p = get_nearest_strike_price(data_list, strike_chosen_s, 'P')

if(is_extreme_c==1):
    needed_c=1
if(is_extreme_p==1):
    needed_p=1

print(f'needed_c {needed_c} needed_p {needed_p}')

interval = 180  # Set the interval in seconds (e.g., 15 or 30 seconds)

next_call = time.time()+interval  # Set the initial time


min_amount=0.4
min_price=0
max_quantity_ini=2
max_quantity=max_quantity_ini
max_quantity_inc_per=30
price_trigger_conf=0
order_response=None
symbol_1=''
atempt=0
current_size=-1
trigger_symbol_per=0.25
close_price=-1
first_time_in=0
tot_loss=0
tot_loss_accepted=min_amount*2.5
intitial_run=0
start_min_amount=0
prof_min_amount=0

print('distance opt chain:')
option_chain_internal_1(data_list,mark_price,-2250,2250)
print('strike touch opt chain:')
option_chain_internal(data_list,mark_price)
condition=False

while(condition):
    try:
        mark_price=float(delta_client.get_ticker(symbol)['mark_price'])
    except RequestException as e:
        print(f"A requests-related error occurred: {e}")
        mark_price=-1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        mark_price=-1


    if(entry_price!=0 and entry_size!=0 and current_size!=0):

        if(intitial_run==0):
          tot_loss_accepted=start_min_amount*2.5 
          print(f'min amount changed from now {min_amount} to {start_min_amount},new tot_loss accepted {tot_loss_accepted}')
          min_amount=start_min_amount
          prof_min_amount=start_min_amount
          

        intitial_run=1

        #print(f'mark pr {mark_price:.2f} entry_price {entry_price} and entry size {entry_size} found not 0')

        if(first_time_in==0):
            strike_price_symbol_1 = float(symbol_1.split('-')[2])
            symbol_type=symbol_1[0]
            ticker_1=delta_client.get_ticker(symbol_1)
            product_id_1=ticker_1['product_id']

            if(symbol_type=='C'):
                trigger_symbol_price=strike_price_symbol_1*(100-trigger_symbol_per)/100
            elif(symbol_type=='P'):
                trigger_symbol_price=strike_price_symbol_1*(100+trigger_symbol_per)/100
        
        first_time_in=1

        print(f'entry_price {entry_price} entry size {entry_size} not 0, mark pr {mark_price:.2f} trigger_pr {trigger_symbol_price}, sym {symbol_1}')
        
        if((symbol_type=='C' and trigger_symbol_price<=mark_price+3000*0)or(symbol_type=='P' and trigger_symbol_price>=mark_price-3000*0) ):
            
            print(f'Placing order entry_price {entry_price} entry size {entry_size} not 0, mark pr {mark_price:.2f} trigger_pr {trigger_symbol_price} sym {symbol_1}')
            try:
                a=1
                #"""
                order_response = delta_client.place_order(
                        product_id=product_id_1,
                        size=entry_size,
                        side='buy',
                        limit_price=strike_price_symbol_1*100,
                        #order_type= "limit_order"
                        #post_only='true',
                        )
                #"""
            except RequestException as e:
                print(f"A requests-related error occurred: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            #"""
            if(order_response is not None):
                close_price=order_response.get("average_fill_price")
                close_size=order_response.get("size")

            #print(order_response)
            print(f'close price {close_price} close size {close_size}')
            #"""

        #symbol_1='C-BTC-55200-090924'
        #quantity_1=2

        if(close_price!=-1):
            try:
                ticker_1=delta_client.get_ticker(symbol_1)
            except RequestException as e:
                print(f"A requests-related error occurred: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            product_id_1=ticker_1['product_id']
            best_bid_1=ticker_1['quotes']['best_bid']
            best_ask_1=ticker_1['quotes']['best_ask']
            close_price=best_ask_1

        try:
            response_order=delta_client.get_position(product_id_1)
        except RequestException as e:
            print(f"A requests-related error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


        current_size=float(response_order.get('size'))

        curr_abs=abs(current_size)*100/100.0
        entr_abs=abs(entry_size)*100/100.0

        if(curr_abs<entr_abs):
            protect_trigger=1

        if(protect_trigger==1 and new_trigger==0):
            
            print(f'protect_triggered, curr_size {curr_abs}, entry_abs {entr_abs}')
            
            size_executed=entr_abs-curr_abs
            loss=(float(close_price)-float(entry_price))
            loss_size=loss*size_executed/1000.0
            tot_loss=tot_loss+loss_size
            #min_amount=min_amount+loss_size
            prof_min_amount_ini=prof_min_amount
            while((prof_min_amount+tot_loss)>prof_min_amount_ini):
                print(f'prof_min_amt {prof_min_amount} tot_loss {tot_loss} prof+tot {prof_min_amount+tot_loss} prof_ini {prof_min_amount_ini} ')
                if(prof_min_amount==0):
                    break
                prof_min_amount=prof_min_amount*90/100
            min_amount=prof_min_amount+tot_loss

            print(f'loss {loss} loss_size {loss_size}, size_executed {size_executed}, atempt {atempt}, max qt {max_quantity}, prof_min_amt {prof_min_amount} tot_loss {tot_loss} min_amt {min_amount} ')

            new_trigger=1
            #response_order=delta_client.get_position(product_id_1)
            #entry_price=
        
        if(protect_trigger==1 and new_trigger==1):
            #atempt=atempt+1
            #max_quantity=max_quantity_ini*(100+max_quantity_inc_per*atempt)/100
            print(f'loss {loss} loss_size {loss_size}, size_executed {size_executed}, atempt {atempt}, max qt {max_quantity}, prof_min_amt {prof_min_amount} tot_loss {tot_loss} min_amt {min_amount} ')

        if(current_size==0):
            price_trigger_conf=1
            if(symbol_1[0]=='C'):
                direction='up'
            elif(symbol_1[0]=='P'):
                direction='down'
            max_quantity=round(max_quantity)
            first_time_in=0
            close_price=-1
            new_trigger=0
            protect_trigger=0
            entry_price=0
            entry_size=0
            current_size=-1
            atempt=atempt+1
            max_quantity=max_quantity_ini*(100+max_quantity_inc_per*atempt)/100
            print(f'loss {loss} loss_size {loss_size}, size_executed {size_executed}, atempt {atempt}, max qt {max_quantity}, prof_min_amt {prof_min_amount} tot_loss {tot_loss} min_amt {min_amount} ')

    
    else:
            
        print(f'mark pr {mark_price:.2f} price_tigger_b {price_trigger_b:.2f} price_tigger_s {price_trigger_s:.2f}')

        if(price_trigger_near_b<=mark_price and needed_c==1 and near_b==0 and mark_price!=-1):

            print(f'price_tigger_near_b {price_trigger_near_b:.2f} ')

            data_list =opt_chain_expiry(expiry_date)
            near_b=1

            strike_chosen_b=price_trigger_b+1000
            nearest_option_info_c, nearest_opt_index_c,is_extreme_c = get_nearest_strike_price(data_list, strike_chosen_b, 'C')

            if(is_extreme_c==1):
                needed_c=1
            
            next_call = time.time()+interval
        
        elif(price_trigger_near_s>=mark_price and needed_p==1 and near_s==0 and mark_price!=-1):

            print(f'price_tigger_near_s {price_trigger_near_s:.2f} ')

            data_list =opt_chain_expiry(expiry_date)
            near_s=1

            strike_chosen_s=price_trigger_s-1000
            nearest_option_info_p, nearest_opt_index_p,is_extreme_p = get_nearest_strike_price(data_list, strike_chosen_s, 'P')

            if(is_extreme_p==1):
                needed_p=1
            
            next_call = time.time()+interval


        if(price_trigger_b<=mark_price and mark_price!=-1):

            print(f'price_tigger_b hit : {price_trigger_b:.2f} ')
            direction='up'
            price_trigger_conf=1
            
        
        if(price_trigger_s>=mark_price and mark_price!=-1):
            print(f'price_tigger_s hit : {price_trigger_s:.2f} ')
            direction='down'
            price_trigger_conf=1
        
        if(price_trigger_conf==1):

            print(f'price_tigger_conf : {price_trigger_conf} direction {direction} ')
            best_option=choose_best_option_with_strike_priority(data_list,mark_price,direction,min_amount,min_price,max_quantity)
            print(f'placing order for {best_option}')

            try:
                a=1
                #"""
                order_response = delta_client.place_order(
                        product_id=best_option['id'],
                        size=best_option['quantity'],
                        side=side,
                        limit_price=0.01,
                        #order_type= "limit_order"
                        #post_only='true',
                        )
                #"""
            except RequestException as e:
                print(f"A requests-related error occurred: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            
            if(order_response is not None):
                entry_price=order_response.get("average_fill_price")
                entry_size=order_response.get("size")
                symbol_1=best_option['symbol']
                print(f'new entry_price {entry_price} entry size {entry_size} sym {symbol_1}')

            #print(order_response)
            price_trigger_conf=0
            price_trigger_b=price_trigger_b*10
            price_trigger_s=price_trigger_s/10
            #condition=False

    if(tot_loss>tot_loss_accepted):
        print(f'tot_loss {tot_loss} crossed tot loss accepted {tot_loss_accepted}, stopping')
        condition=False
        
    current_time = time.time()
    if current_time >= next_call:
        my_function()
        near_b=0
        near_s=0
        next_call = current_time + interval  # Set the next call time