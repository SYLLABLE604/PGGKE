import geopandas as gp
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
# from langchain_google_vertexai import VertexAI
from langchain.output_parsers import CommaSeparatedListOutputParser,PydanticOutputParser
from langchain_core.messages import HumanMessage
import os
from getpass import getpass
import json
from langchain.prompts import PromptTemplate
from difflib import SequenceMatcher
from langchain_core.pydantic_v1 import BaseModel, Field
import ast
import base64
import time
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from itertools import permutations,combinations
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString,Point

def min_max_range(x, range_values):
    return [round( ((xx - min(x)) / (1.0*(max(x) - min(x)))) * (range_values[1] - range_values[0]) + range_values[0], 2) for xx in x]

def draw_origional_map(node_data,edge_index,unselect_location,location_center_point,county,num):
    #待分配位置
    x = unselect_location['Longitude'].to_numpy()
    y = unselect_location['Latitude'].to_numpy()
    
    colors = []
    for i,row in unselect_location.iterrows():
        if row['Node Type'] == 'PQ':
            colors.append('green')
        elif row['Node Type'] == 'PV':
            colors.append('orange')
    
    setted_colors = []
    for i,row in node_data.iterrows():
        if row['Type'] == 0:
            setted_colors.append('blue')
        else:
            setted_colors.append('red')

    #初始草图
    setted_x = node_data['X'].to_numpy()
    setted_y = node_data['Y'].to_numpy()
    setted_label = node_data['ID'].to_numpy()
    # setted_colors = []
    # for i in range(len(setted_label)):
    #     setted_colors.append('orange')

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x, y, s=10, c=colors)
    plt.scatter(location_center_point[0], location_center_point[1], s=50, c='yellow')
    plt.scatter(setted_x, setted_y, s=30, c=setted_colors)
    for i, label in enumerate(setted_label):
        ax.annotate(label, (setted_x[i], setted_y[i]), fontsize=14, xytext=(5, 5), textcoords="offset points")
    
    for i,row in edge_index.iterrows():
        # print(i)
        start_index = node_data[node_data['ID'] == row['Start']].index
        end_index = node_data[node_data['ID'] == row['End']].index
        x1, x2 = setted_x[start_index],setted_x[end_index]
        y1, y2 = setted_y[start_index],setted_y[end_index]
        plt.plot([x1,x2],[y1,y2],'k-')

    ax.axis('off')
    # 添加标题和坐标轴标签
    plt.title('Locations on a Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # 显示网格线
    plt.grid(True)

    # 指定保存路径
    save_path = f'frp/geo/Middle step/process image/draft {county} {num}.png'

    # 保存图像
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def draw_normal_map(node_data,target_PV_location,location_center_point,county,number):#先画出现有节点的连线图，然后把wait_selection点画上去
    #位置
    x_combination = target_PV_location['Longitude'].to_numpy()
    y_combination = target_PV_location['Latitude'].to_numpy()
    x_draft = node_data['X'].to_numpy()
    y_draft = node_data['Y'].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_combination, y_combination, s=50, c='orange')
    ax.scatter(x_draft, y_draft, s=50, c='blue')
    ax.scatter(location_center_point[0], location_center_point[1], s=50, c='red')

    setted_label = node_data['ID'].to_numpy()
    for i, label in enumerate(setted_label):
        ax.annotate(label, (x_draft[i], y_draft[i]), xytext=(5, 5), textcoords="offset points")
    setted_label_combination = target_PV_location['Zip'].to_numpy()
    for i, label in enumerate(setted_label_combination):
        ax.annotate(label, (x_combination[i], y_combination[i]), xytext=(5, 5), textcoords="offset points")
    
    ax.axis('off')
    # 添加标题和坐标轴标签
    ax.set_title('Locations on a Map')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 指定保存路径
    save_path = f'frp/geo/Middle step/process image/locations_map {county} {number}.png'

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def get_geo(edge_data, node_data):
    #选择county
    grid_landuse = ''
    G = nx.from_edgelist(edge_data[["Start","End"]].values)
    diameter = nx.diameter(G)
    shortest_path = nx.average_shortest_path_length(G)
    try:
        cycles = nx.cycle_basis(G)
        cycle_number = len(cycles)
        cycle_len = 0
        for j in range(cycle_number):
            cycle_len = max(len(cycles[j]),cycle_len)
        # grid_landuse = get_tag(cycle_len,cycle_number,diameter,shortest_path)
        grid_landuse = 'city'
    except:#无环的话，设置为rural
        grid_landuse = 'rural'
    print(grid_landuse)

    trans_node = pd.read_csv("~/frp/geo/trans_all.csv")
    trans_PQ_node = trans_node[trans_node['Node Type']=="PQ"]

    generate_trans_node_list = pd.DataFrame(columns=['Longitude','Latitude','Node Type','Population','Province','Land Use','Generator Num','Load Num'])

    max_sub_grid_load_num = node_data[node_data['Type'] == 0].count()['ID'] * 1.5 #这里load个数
    max_sub_grid_gene_num = node_data[node_data['Type'].isin([1,2])].count()['ID']
    # print(max_sub_grid_load_num)

    wait_selection_county = trans_PQ_node[(trans_PQ_node['Generator Num'] > max_sub_grid_gene_num) & (trans_PQ_node['Load Num'] > max_sub_grid_load_num)]
    trans_load_rural = wait_selection_county[(wait_selection_county["Land Use"]=="rural")].reset_index()
    trans_load_city = wait_selection_county[(wait_selection_county["Land Use"]=="city")].reset_index()
    trans_load_industry = wait_selection_county[(wait_selection_county["Land Use"]=="industry")].reset_index()

    target_county_list = []
    img_path_list = []
    node_geo_list = []
    new_node = node_data
    for p in range(3):
        if p > 0:
            img_path_list.append(geojson_map_path)
            img_path_list.append(geojson_map_path)
            target_county_list.append(target_county['Province'])
            target_county_list.append(target_county['Province'])
            best_image_num,county_name = LLM_check(img_path_list,target_county_list)
            return node_geo_list[best_image_num-1],county_name

        while(True):
            if(grid_landuse == 'rural'):
                target_county = trans_load_rural.iloc[np.random.randint(0,len(trans_load_rural)-1)]
                # print(trans_load_rural)
            elif(grid_landuse == 'city'):
                target_county = trans_load_city.iloc[np.random.randint(0,len(trans_load_city)-1)]
                # print(trans_load_city)
            elif(grid_landuse == 'industry'):
                target_county = trans_load_industry.iloc[np.random.randint(0,len(trans_load_industry)-1)]
            if(target_county['Province'] not in target_county_list):
                break
        # print(wait_selection_county)
        target_county = wait_selection_county[wait_selection_county['Province'] == 'San Diego'].iloc[0]
        target_county_list.append(target_county['Province'])
        print(target_county)

        #分配网选点
        dist_node = pd.read_csv("frp/geo/dist_all.csv")
        target_county_node = dist_node[dist_node['Province'] == target_county['Province']]
        dist_PQ_node = target_county_node[target_county_node['Node Type']=="PQ"]
        dist_PV_node = target_county_node[target_county_node['Node Type']=="PV"]

        sub_grid_feature = []
        PV_count = 0
        PQ_count = 0
        for j, row2 in new_node.iterrows():
            if(row2['Type'] == 0):
                PQ_count += 1
            elif(row2['Type'] in [1,2]):
                PV_count += 1
        sub_grid_feature.append(PQ_count)
        sub_grid_feature.append(PV_count)
        # print(sub_grid_feature) #每个子网PQ，PV节点个数

        #将点位映射county范围内的真实坐标系
        G = nx.from_edgelist(np.array(edge_data)[:,[0,1]])
        position = nx.kamada_kawai_layout(G)
        position_list = pd.DataFrame.from_dict(position,orient='index',columns=['X','Y'])
        position_list.sort_index(inplace=True)
        position_list = position_list.reset_index()
        new_node['X'] = position_list['X']
        new_node['Y'] = position_list['Y']

        # location_X = target_county_node['Longitude']
        # location_Y = target_county_node['Latitude']
        location_center_point = [dist_PV_node['Longitude'].sum()/len(dist_PV_node),dist_PV_node['Latitude'].sum()/len(dist_PV_node)]
        distance = 0
        for i,row in target_county_node.iterrows():
            distance += pow((row['Longitude'] - location_center_point[0])**2 + (row['Latitude'] - location_center_point[1])**2,0.5)
        average_distance = distance/len(target_county_node)

        max_gauss = np.random.normal(0,0.05)
        min_gauss = np.random.normal(0,0.05)
        draft_X = min_max_range(new_node['X'].values.tolist(),((location_center_point[0]-average_distance)+min_gauss,(location_center_point[0]+average_distance)+max_gauss))
        draft_Y = min_max_range(new_node['Y'].values.tolist(),((location_center_point[1]-average_distance)+min_gauss,(location_center_point[1]+average_distance)+max_gauss))
        # draft_X = min_max_range(new_node['X'].values.tolist(),(location_X.min(),location_X.max()))
        # draft_Y = min_max_range(new_node['Y'].values.tolist(),(location_Y.min(),location_Y.max()))
        # draft_X = min_max_range(new_node['X'].values.tolist(),((location_center_point[0]-average_distance)-0.01,(location_center_point[0]+average_distance)+0.01))
        # draft_Y = min_max_range(new_node['Y'].values.tolist(),((location_center_point[1]-average_distance)-0.01,(location_center_point[1]+average_distance)+0.01))
    
        #按照位置的聚合程度进行放缩，让草图落于聚落中心位置
        new_node['X'] = draft_X
        new_node['Y'] = draft_Y
        # print(new_node)
        draft_center_point = [new_node['X'].sum()/len(new_node),new_node['Y'].sum()/len(new_node)]
        X = 0
        Y = 0
        population_sum = target_county_node['Population'].sum()
        for i,row in target_county_node.iterrows():
            X += row['Longitude']*row['Population']/population_sum
            Y += row['Latitude']*row['Population']/population_sum
        center_random_X = np.random.normal(0,0.05)
        center_random_Y = np.random.normal(0,0.05)
        location_center_point = [X+center_random_X,Y+center_random_Y]
        print(location_center_point)
        # location_center_point = [-116.9860881922366,32.84874033257483]
        # print(draft_center_point)

        #平移整体图到中心重合，并旋转图使PV节点接近目标位置接近
        distance_X = location_center_point[0] - draft_center_point[0]
        distance_Y = location_center_point[1] - draft_center_point[1]
        new_node['X'] = new_node['X'] + distance_X
        new_node['Y'] = new_node['Y'] + distance_Y
        # print(new_node)
        

        # print(target_PV_location)
        
        #画图，橙色draft，蓝色combination，红色中心点
        draw_origional_map(new_node,edge_data,target_county_node,location_center_point,target_county['Province'], 1)
        # draw_normal_map(node_data_PV,dist_PV_node,location_center_point,target_county['Province'], 1)

        #寻找最适合的角度，旋转图
        minimal_sum_distance = 1000
        minimal_group_angle = 0
        
        location_target = []
        trans_label = 0

        for s in range(1):
            angle = 0
            mirror_new_node = new_node
            # print(location_center_point[0])
            # print(new_PV_node_data['X'])

            if s == 1:
                mirror_new_node['X'] = mirror_new_node['X'].map(lambda x: 2 * location_center_point[0] - x)
                # print(new_node_1)
            new_PV_node_data = mirror_new_node[mirror_new_node['Type'] != 0]

            while(angle < 360):
                location_set = []
                distance_sum = 0
                
                for i,row in new_PV_node_data.iterrows():
                    target = 0
                    minimal_distance = 100
                    for j,row2 in dist_PV_node.iterrows():
                        distance = pow((row['X'] - row2['Longitude'])**2 + (row['Y'] - row2['Latitude'])**2,0.5)
                        if (distance < minimal_distance) and (row2['Zip'] not in location_set):
                            minimal_distance = distance
                            target = row2['Zip']
                            # print(target)
                    distance_sum += minimal_distance
                    location_set.append(target)
                # print(distance_sum)
                if distance_sum < minimal_sum_distance:
                    target_PV_location = dist_PV_node[dist_PV_node['Zip'].isin(location_target)]
                    minimal_sum_distance = distance_sum
                    minimal_group_angle = angle
                    location_target = location_set
                    trans_label = s
                    print(s)
                    print(location_target)
                    print(distance_sum)
                    print(minimal_group_angle)

                angle += 1
                angle_trans = math.radians(1)
                x0 = (new_PV_node_data['X']-location_center_point[0])*math.cos(angle_trans) + (new_PV_node_data['Y']-location_center_point[1])*math.sin(angle_trans) + location_center_point[0]; 
                y0 = -(new_PV_node_data['X']-location_center_point[0])*math.sin(angle_trans) + (new_PV_node_data['Y']-location_center_point[1])*math.cos(angle_trans) + location_center_point[1]; 

                new_PV_node_data.loc[:,'X'] = x0
                new_PV_node_data.loc[:,'Y'] = y0

        target_PV_location = dist_PV_node[dist_PV_node['Zip'].isin(location_target)]
        # location_target += [0]*(len(new_node)-len(location_target))
        location_list = [0] * len(new_node)
        m = 0
        # print(location_target)
        for i,row in new_PV_node_data.iterrows():
            location_list[int(row['ID'])-1] = location_target[m]
            m += 1
        print(location_list)
        
        # print(trans_label)
        # print(minimal_group_angle)
        if trans_label == 1:
            new_node = mirror_new_node
            print("mirror transition ")
        # print(new_node)

        new_node.loc[:,'target'] = location_list
        print(new_node)

        draw_origional_map(new_node,edge_data,target_county_node,location_center_point,target_county['Province'],2)
        angle_trans = math.radians(minimal_group_angle)
        x0 = (new_node['X']-location_center_point[0])*math.cos(angle_trans) + (new_node['Y']-location_center_point[1])*math.sin(angle_trans) + location_center_point[0]; 
        y0 = -(new_node['X']-location_center_point[0])*math.sin(angle_trans) + (new_node['Y']-location_center_point[1])*math.cos(angle_trans) + location_center_point[1]; 
        new_node.loc[:,'X'] = x0
        new_node.loc[:,'Y'] = y0
        print(new_node)
        # print(new_node)
        draw_origional_map(new_node,edge_data,target_county_node,location_center_point,target_county['Province'],3)
        # draw_normal_map(new_node[new_node['Type'] != 0],target_PV_location,location_center_point,target_county['Province'],2)
        
        #旋转完成，设置位移变量，平移PV节点
        node_data_PV = new_node[new_node['Type'] != 0]
        translation_vector = pd.DataFrame(columns=['ID','distance_X','distance_Y'])
        for i,row in node_data_PV.iterrows():
            new_node.loc[i,'X'] = target_PV_location[target_PV_location['Zip'] == row['target']]['Longitude'].values.tolist()
            new_node.loc[i,'Y'] = target_PV_location[target_PV_location['Zip'] == row['target']]['Latitude'].values.tolist()
            target_location = target_PV_location[target_PV_location['Zip'] == row['target']]
            group = []
            distance_X = target_location['Longitude'].values.tolist()[0] - row['X']
            distance_Y = target_location['Latitude'].values.tolist()[0] - row['Y']
            group.append(row["ID"])
            group.append(distance_X)
            group.append(distance_Y)
            new_row = pd.DataFrame([group], columns=translation_vector.columns)
            translation_vector = pd.concat([translation_vector.dropna(axis=1, how='all'), new_row], ignore_index=True)
        # print(translation_vector)
        # print(node_data_PV)
        # print(new_node)
        
        #初始化每个节点的移动矩阵
        node_move_matrix = pd.DataFrame(columns=['ID','distance_X','distance_Y'])
        assigned_node = new_node[new_node['target'] != 0]
        unassigned_node = new_node[new_node['target'] == 0]
        #计算每个节点到出发点的步数，加权平移
        for i, row in unassigned_node.iterrows():
            distance_X = 0
            distance_Y = 0
            distance_sum = 0
            distance_group = pd.DataFrame(columns=['neighbor_ID','source','step'])
            for j, row2 in assigned_node.iterrows():
                distance = nx.shortest_path_length(G, source=row['ID'], target=row2['ID'])
                data = [row['ID'],row2['ID'],distance]
                new_row = pd.DataFrame([data], columns=distance_group.columns)
                distance_group = pd.concat([distance_group.dropna(axis=1, how='all'), new_row], ignore_index=True)
            for o,row4 in distance_group.iterrows():
                distance_sum += (1/row4['step'])
            for k, row3 in distance_group.iterrows():
                distance_X += ((1/row3['step'])/distance_sum)*(translation_vector[translation_vector['ID'] == row3['source']]['distance_X'].values.tolist()[0])
                distance_Y += ((1/row3['step'])/distance_sum)*(translation_vector[translation_vector['ID'] == row3['source']]['distance_Y'].values.tolist()[0])
            new_row = pd.DataFrame([[row['ID'],distance_X,distance_Y]], columns=node_move_matrix.columns)
            node_move_matrix = pd.concat([node_move_matrix.dropna(axis=1, how='all'), new_row], ignore_index=True) 
        
        #所有非PV节点进行移动
        # node_move_matrix = node_move_matrix.set_index(pd.Index(range(len(assigned_node),len(new_node))))
        for i,row in node_move_matrix.iterrows():
            new_node.loc[new_node[new_node['ID'] == row['ID']].index,'X'] = unassigned_node[unassigned_node['ID'] == row['ID']]['X'] + row['distance_X']
            new_node.loc[new_node[new_node['ID'] == row['ID']].index,'Y'] = unassigned_node[unassigned_node['ID'] == row['ID']]['Y'] + row['distance_Y']
        # new_node.loc[len(assigned_node):,'X'] = unassigned_node['X'] + node_move_matrix['distance_X']
        # new_node.loc[len(assigned_node):,'Y'] = unassigned_node['Y'] + node_move_matrix['distance_Y']
        # print(assigned_node)
        # print(new_node)
        draw_origional_map(new_node,edge_data,target_county_node,location_center_point,target_county['Province'],4)

        #寻找邻居节点,迭代搜索
        a = 0
        assigned_last_round_node = assigned_node
        unassigned_node = new_node[new_node['target'] == 0]
        while len(assigned_node) != len(new_node):
            #寻找邻居节点
            wait_assigned_neighbor = pd.DataFrame(columns = new_node.columns)
            for i,row in assigned_last_round_node.iterrows():
                for j,row2 in edge_data.iterrows():
                    if (row['ID'] == row2['Start']) and (row2['End'] not in assigned_node['ID'].values.tolist()):
                        neighbor_node = unassigned_node[unassigned_node['ID'] == row2['End']]
                        wait_assigned_neighbor = pd.concat([wait_assigned_neighbor.dropna(axis=1, how='all'), neighbor_node], ignore_index=True)
                    elif (row['ID'] == row2['End']) and (row2['Start'] not in assigned_node['ID'].values.tolist()):
                        neighbor_node = unassigned_node[unassigned_node['ID'] == row2['Start']]
                        wait_assigned_neighbor = pd.concat([wait_assigned_neighbor.dropna(axis=1, how='all'), neighbor_node], ignore_index=True)
            wait_assigned_neighbor = wait_assigned_neighbor.drop_duplicates().reset_index(drop=True)
            # print(new_node)
            #已经被分配过的load zip
            assigned_list = assigned_node[assigned_node['Type'] == 0]['target'].values.tolist()

            translation_vector = pd.DataFrame(columns=['ID','distance_X','distance_Y'])
            for i,row in wait_assigned_neighbor.iterrows():
                minimal_distance = 100
                for j,row2 in dist_PQ_node.iterrows():
                    distance = pow((row['X'] - row2['Longitude'])**2 + (row['Y'] - row2['Latitude'])**2,0.5)
                    if (distance < minimal_distance) and (row2['Zip'] not in assigned_list):
                        minimal_distance = distance
                        x = row2['Longitude']
                        y = row2['Latitude']
                        target = row2['Zip']
                assigned_list.append(target)
            
                # new_node[new_node['ID'] == row['ID']]['target'] = target
                new_node.loc[new_node[new_node['ID'] == row['ID']].index,'target'] = target
                new_node.loc[new_node[new_node['ID'] == row['ID']].index,'X'] = x
                new_node.loc[new_node[new_node['ID'] == row['ID']].index,'Y'] = y
                
                #计算这一轮节点的位移量
                group = []
                group.append(row["ID"])
                group.append(x - row['X'])
                group.append(y - row['Y'])
                new_row = pd.DataFrame([group], columns=translation_vector.columns)
                translation_vector = pd.concat([translation_vector.dropna(axis=1, how='all'), new_row], ignore_index=True)
            # print(wait_assigned_neighbor)
            # print(new_node[new_node['ID'].isin(wait_assigned_neighbor['ID'].values.tolist())])
            # print(translation_vector)

            assigned_node = new_node[new_node['target'] != 0]
            unassigned_node = new_node[new_node['target'] == 0]

            #计算其他未分配节点的移动变量
            node_move_matrix = pd.DataFrame(columns=['ID','distance_X','distance_Y','number'])
            for i, row in unassigned_node.iterrows():
                distance_X = 0
                distance_Y = 0
                distance_sum = 0
                distance_group = pd.DataFrame(columns=['neighbor_ID','source','step'])
                for j, row2 in wait_assigned_neighbor.iterrows():
                    distance = nx.shortest_path_length(G, source=row['ID'], target=row2['ID'])
                    data = [row['ID'],row2['ID'],distance]
                    new_row = pd.DataFrame([data], columns=distance_group.columns)
                    distance_group = pd.concat([distance_group.dropna(axis=1, how='all'), new_row], ignore_index=True)
                for o,row4 in distance_group.iterrows():
                    distance_sum += (1/1+row4['step'])
                for k, row3 in distance_group.iterrows():
                    distance_X += ((1/1+row3['step'])/distance_sum)*(translation_vector[translation_vector['ID'] == row3['source']]['distance_X'].values.tolist()[0])
                    distance_Y += ((1/1+row3['step'])/distance_sum)*(translation_vector[translation_vector['ID'] == row3['source']]['distance_Y'].values.tolist()[0])
                new_row = pd.DataFrame([[row['ID'],distance_X,distance_Y,i]], columns=node_move_matrix.columns)
                node_move_matrix = pd.concat([node_move_matrix.dropna(axis=1, how='all'), new_row], ignore_index=True) 

            node_move_matrix = node_move_matrix.set_index('number')
            # print(node_move_matrix)
            for i,row in node_move_matrix.iterrows():
                new_node.loc[i,'X'] = new_node.loc[i,'X'] + row['distance_X']
                new_node.loc[i,'Y'] = new_node.loc[i,'Y'] + row['distance_Y']

            a += 1
            assigned_last_round_node = wait_assigned_neighbor
            print(new_node)
            draw_origional_map(new_node,edge_data,target_county_node,location_center_point,target_county['Province'],4+a)
        print(new_node)

        # draw_final_map(new_node,edge_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"node_location_of_{str(len(node_data))}_{target_county['Province']}"
        node_geo_path = f"frp/geo/Middle step/node_location/{file_name}.csv"
        geojson_map_path = f"frp/geo/Middle step/geojson_data/{file_name}_{timestamp}.png"

        new_node.to_csv(node_geo_path,index=False)
        draw_geojson_map(new_node,edge_data,geojson_map_path)

        node_geo_list.append(node_geo_path)
        img_path_list.append(geojson_map_path)

    print(node_geo_list)
    #大模型检验哪个图更好
    best_image_num,county_name = LLM_check(img_path_list,target_county_list)
    
    return node_geo_list[best_image_num-1],county_name

def draw_geojson_map(node_data,edge_data,geojson_map_path):
    
    dist_PV_node = node_data[node_data['Type'].isin([1,2])]
    dist_PQ_node = node_data[node_data['Type'] == 0]
    dist_PV_geodataframe_node = gpd.GeoDataFrame(dist_PV_node, geometry=gpd.points_from_xy(dist_PV_node.X, dist_PV_node.Y))
    dist_PQ_geodataframe_node = gpd.GeoDataFrame(dist_PQ_node, geometry=gpd.points_from_xy(dist_PQ_node.X, dist_PQ_node.Y))

    # geodataframe_node.to_file(os.path.join(OUTPUT_DIR, 'node_location_of_25_20240715_091137.geojson'), driver='GeoJSON')
    tb = []
    geomList = []
    Max_Longitude = node_data['X'].max()
    Max_Latitude = node_data['Y'].max()
    Min_Longitude = node_data['X'].min()
    Min_Latitude = node_data['Y'].min()

    for i,row in edge_data.iterrows():
        # 分离出属性信息，取每行后2列作为数据属性
        tb.append(row.iloc[2:])
        Longitude = []
        Latitude = []
        Longitude.append(node_data[node_data['ID'] == row['Start']]['X'])
        Longitude.append(node_data[node_data['ID'] == row['End']]['X'])
        Latitude.append(node_data[node_data['ID'] == row['Start']]['Y'])
        Latitude.append(node_data[node_data['ID'] == row['End']]['Y'])

        xyList = [xy for xy in zip(Longitude, Latitude)]
        line = LineString(xyList)
        geomList.append(line)
    rest_geodataframe_edge = gpd.GeoDataFrame(tb, geometry = geomList)

    county = gpd.read_file("frp/geo/shapefile/Zip/cb_2018_us_zcta510_500k.shp")
    # county.to_crs(epsg=4326,inplace=True)
    fig, ax = plt.subplots(figsize = (10,15))
    plt.xlim((Min_Longitude-0.05, Max_Longitude + 0.05))
    plt.ylim((Min_Latitude-0.05, Max_Latitude + 0.05))

    county.plot(ax=ax,color = "grey")
    dist_PV_geodataframe_node.plot(ax=ax, color= 'red')
    dist_PQ_geodataframe_node.plot(ax=ax, color = 'blue')
    rest_geodataframe_edge.plot(ax=ax, color = 'black')
    plt.savefig(geojson_map_path)
    plt.close()

def draw_final_map(node_geo_sum,edge_data,name):#先画出现有节点的连线图，然后把wait_selection点画上去
    #主网
    x = node_geo_sum['X'].to_numpy()
    y = node_geo_sum['Y'].to_numpy()
    labels = []
    for i in range(len(x)):
        labels.append(node_geo_sum['ID'].iloc[i])
    colors = []
    for i,row in node_geo_sum.iterrows():
        if row['Type'] == 0:
            colors.append('blue')
        elif row['Type'] in [1,2]:
            colors.append('red')

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.scatter(x, y, s=10, c=colors)
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords="offset points")
    
    #将已分配节点进行连线
    for i,row in edge_data.iterrows():
        start_index = node_geo_sum[node_geo_sum['ID'] == row['Start']].index
        end_index = node_geo_sum[node_geo_sum['ID'] == row['End']].index
        x1, x2 = x[start_index],x[end_index]
        y1, y2 = y[start_index],y[end_index]
        plt.plot([x1,x2],[y1,y2],linewidth=.2,color ='black')

    ax.axis('off')
    # 添加标题和坐标轴标签
    plt.title('Grid Structure on a Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # 显示网格线
    plt.grid(True)

    # 指定保存路径
    save_path = f'frp/geo/Output/Image/{name}.png'

    # 保存图像
    plt.savefig(save_path)
    return save_path

def LLM_check(image_path_list,target_county_list):
    image_data_list = []
    for i in range(len(image_path_list)):
        with open(image_path_list[i], "rb") as image_file:
    # 将图像文件读取为二进制数据
            image_data = image_file.read()
            image_data = base64.b64encode(image_data).decode("utf-8")
        image_data_list.append(image_data)
        
    
    # llm = VertexAI(model_name="gemini-pro-vision")
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#     llm = AzureChatOpenAI(
#         azure_deployment="GPT4o",
#         api_version="2024-04-01-preview",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         # other params...
#     )
    
#     message = HumanMessage(
#         content=[
#             {"type": "text", 
#             "text": f"""
# Your task is to verify the quality of these three power system grid image. You need to output the score of each image under the following standard:

# Node Clarity and Distribution: (full score 5):
#       5 points: Nodes are clearly visible, well-distributed, and evenly spaced.
#       4 points: Nodes are mostly clear with minor clustering.
#       3 points: Some nodes are clustered or overlapping.
#       2 points: Significant clustering or overlap of nodes.
#       1 point: Nodes are largely indistinguishable or overlapping.

# Edge Clarity and Distinctness: (full score 5):
#       5 points: Edges are clear, distinct, and easy to follow.
#       4 points: Most edges are clear with minor overlap.
#       3 points: Some edges are difficult to distinguish due to overlap.
#       2 points: Many edges overlap or are unclear.
#       1 point: Edges are mostly overlapping and indistinct.

# Line Crossing and Intersections (full score 5):
#       5 points: Minimal or no line crossings.
#       4 points: A few crossings that do not affect clarity.
#       3 points: Noticeable crossings causing some confusion.
#       2 points: Frequent crossings reducing clarity.
#       1 point: Excessive crossings making the image very unclear.

# Graph structure (full score 5):
#       5 points: All of the junctions are obtuse angles
#       4 points: Most of the junctions are obtuse angles
#       3 points: Some of the junctions are acute angle
#       2 points: Most of junctions are acute angle.
#       1 point: All of junctions are acute angle.

# The full score is 20, please output score and the reason of each 3 images from above four perspectives.

# Formate:
# Image 1 Score: {{total score}}
#       Node Clarity and Distribution: {{score}} {{reason}}
#       Edge Clarity and Distinctness: {{score}} {{reason}}
#       Line Crossing and Intersections: {{score}} {{reason}}
#       Graph structure: {{score}} {{reason}}
# Image 2 Score: {{total score}}
#       Node Clarity and Distribution: {{score}} {{reason}}
#       Edge Clarity and Distinctness: {{score}} {{reason}}
#       ......
#     """},
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{image_data_list[0]}"},
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{image_data_list[1]}"},
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{image_data_list[2]}"},
#                 },
#             ],
#         )
#     response1 = llm.invoke([message]).content
#     print("original output:")
#     print(response1)
    
#     while(True):
#         parser = PydanticOutputParser(pydantic_object=Answer)
#         prompt = PromptTemplate(
#             template="""Your task is to summarize the output from the response generated by model and output the best image name.
#       1, Select the image with highest score. 
#       2, If the input says all of the image are of same quality, random select one as output.
#       3, The score of best image should be higher than 15, if not, return 0.
#       4, The imput should include scores of 3 images, if not, return
#     input:{question}\n {format_instructions}\n """,
#             input_variables=["question"],
#             partial_variables={"format_instructions": parser.get_format_instructions()},
#         )
#         llm = ChatGoogleGenerativeAI(model="gemini-pro")
#         _input = prompt.format_prompt(question=response1)
#         output = llm(_input.to_string())
#         output = ast.literal_eval(output)["tag"]
#         print(output)
#         if(int(output) in [1,2,3]):
#             break
    output = 1
    return int(output),target_county_list[int(output)-1]

def save_final_data(node_geo_sum,edge_data,county_name,timestamp):
    num = len(node_geo_sum)
    name = f'{county_name} {num} bus {timestamp}'
    # 最后删除
    background_path = f"frp/geo/Output/Background/background {name}.png"
    unassigned_node_path = f"frp/geo/Output/Wait_assigned_node/alternative {name}.json"
    edge_data_path = f"frp/geo/Output/edge_data/edge {name}.csv"
    node_path = f"frp/geo/Output/node_data/node {name}.csv"
    json_file_path = f'frp/geo/Output/Json/{name}.json'

    unassigned_node = pd.read_csv("frp/geo/dist_all.csv")
    Max_Longitude = node_geo_sum['X'].max()
    Max_Latitude = node_geo_sum['Y'].max()
    Min_Longitude = node_geo_sum['X'].min()
    Min_Latitude = node_geo_sum['Y'].min()
    unassigned_node = unassigned_node[unassigned_node['Province'] == county_name]
    # unassigned_node = unassigned_node[(unassigned_node['Longitude'] < Max_Longitude) & (unassigned_node['Longitude'] > Min_Longitude) & (unassigned_node['Latitude'] < Max_Latitude) & (unassigned_node['Latitude'] > Min_Latitude)]
    
    # county = gpd.read_file("frp/geo/shapefile/Zip/cb_2018_us_zcta510_500k.shp")
    # # county.to_crs(epsg=4326,inplace=True)
    # fig, ax = plt.subplots(figsize = (10,15))
    # plt.xlim((Min_Longitude-0.05, Max_Longitude + 0.05))
    # plt.ylim((Min_Latitude-0.05, Max_Latitude + 0.05))
    # county.plot(ax=ax,color = "grey")
    # plt.savefig(background_path)
    # plt.close(node_path)

    # unassigned_node.to_csv(unassigned_node_path,index=False)
    # node_geo_sum.to_csv(node_path,index=False)
    # edge_data.to_csv(edge_data_path,index=False)
    
    key1 = 'connections'
    key2 = 'points'
    
    format_spec1 = {
        'Start': 'Start',
        'End': 'End',
        'Edge type': 'Edge type'
    }
    format_spec2 = {
        'ID': 'ID',
        'Type': 'Type',
        'X': 'X',
        'Y': 'Y',
        'Zip': 'target'
    }
    
    # 格式化数据
    formatted_data1 = edge_data.rename(columns=format_spec1).to_dict(orient='records')
    formatted_data2 = node_geo_sum.rename(columns=format_spec2).to_dict(orient='records')
    
    # 合并数据
    merged_data = {
        key1: formatted_data1,
        key2: formatted_data2
    }
    
    # 写入JSON文件
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(merged_data, json_file, indent=4, ensure_ascii=False)
    

    key1 = 'points'
    # json_file_path = unassigned_node_path
    format_spec1 = {
        'Longitude': 'Longitude',
        'Latitude': 'Latitude',
        'Node Type': 'Node_Type',
        'Population': 'Population',
        'Zip': 'Zip',
        'Province': 'Province'
    }

    # 格式化数据
    formatted_data1 = unassigned_node.rename(columns=format_spec1).to_dict(orient='records')
    merged_data = {
        key1: formatted_data1
    }

    # 写入JSON文件
    with open(unassigned_node_path, mode='w', encoding='utf-8') as json_file:
        json.dump(merged_data, json_file, indent=4, ensure_ascii=False)

    path_list = []
    path_list.append(json_file_path)
    path_list.append(unassigned_node_path)

    return path_list
    
class Answer(BaseModel):
    tag: str = Field(description="1,2,3")

# def merge_csv_to_json(name):
#     csv_file1 = f'test code/advance function/geodrawing/Distribution network/Output/edge_data/edge {name}.csv'
#     csv_file2 = f'test code/advance function/geodrawing/Distribution network/Output/node_data/node {name}.csv'
    
def main(edge_path,node_path):
    # name = "29_20240906_113628"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    edge_data = pd.read_csv(edge_path)
    node_data = pd.read_csv(node_path)
    node_geo_path,county_name = get_geo(edge_data,node_data)

    #save the output
    node_geo_sum = pd.read_csv(node_geo_path)
    path_list = save_final_data(node_geo_sum,edge_data,county_name,timestamp)

    connections_json_path = path_list[0] #提取path_list的args对吗？
    alternative_json_path = path_list[1]

    print(f"The connections is {connections_json_path}")
    print(f"The  alternative is {alternative_json_path}")

    with open(connections_json_path, 'r', encoding='utf-8') as f:
        connections_data = json.load(f)

    with open(alternative_json_path, 'r', encoding='utf-8') as f:
        alternative_data = json.load(f)

    print(connections_data)
    print(alternative_data)
    return connections_data, alternative_data

    # return path_list


# main("frp/geo/edge.csv","frp/geo/node.csv")
# for i in range(10):
# main("frp/geo/new_edge_San_Diego_bus.csv","frp/geo/new_node_San_Diego_bus.csv")

# #转换为graphml格式文件
# G = nx.from_edgelist(edge_data[["Start","End"]].values)
# position_dict = node_geo_sum.set_index('ID')[['X', 'Y']].to_dict('index')
# # print(position_dict)

# nx.set_node_attributes(G,position_dict,'position')
# color = []
# for i, row in node_geo_sum.iterrows():
#     if(row['Type'] == 0):
#         color.append('black')
#     elif(row['Type'] in [1,2]):
#         color.append('red')
# node_geo_sum['color'] = color

# color_dict = node_geo_sum.set_index('ID')['color'].to_dict()
# # print(color_dict)
# nx.set_node_attributes(G,color_dict,'color')

# color = nx.get_node_attributes(G, "color")
# pos = nx.get_node_attributes(G, "pos")
# # print(color)
# # print(pos)

# nx.write_graphml(G, f'test code/advance function/geodrawing/Distribution network/Output/graphml data/{17_20240808_073326}.graphml')


