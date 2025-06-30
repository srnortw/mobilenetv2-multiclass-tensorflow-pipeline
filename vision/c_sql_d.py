# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:14:20 2024

@author: Serkan
"""
import zipfile

import pdb

import psycopg2 as pg2

import argparse

from dotenv import load_dotenv
import os

import gdown

import pandas as pd
#pdb.set_trace()

#from concurrent.futures import ThreadPoolExecutor

class c_sql_d_c():
    def __init__(self,host,port,database,user,password):
        
        self.host=host
        self.port=port
        self.database=database
        self.user=user
        self.password=password
        self.conn=pg2.connect(host=self.host,port=self.port,database=self.database,user=self.user,password=self.password)
        self.cur=self.conn.cursor()
        


    def query(self,sqlcommand):
        # Execute the SQL command to create the table)
        self.cur.execute(sqlcommand)
        #bring
        keys=[desc[0] for desc in self.cur.description]
        li=self.cur.fetchall()
        return pd.DataFrame(li,columns=keys)

    def process(self,sqlcommand):
        # Execute the SQL command to create the table)
        self.cur.execute(sqlcommand)
        # Commit the transaction
        self.conn.commit()
    
    def close_connection(self):
        # Close cursor and connection
        self.cur.close()
        self.conn.close()
        


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(
        description='put datasets into sql database')
    
    parser.add_argument(
        '-dn',
        '--dataset_name')
    
    parser.add_argument(
        '-dsn',
        '--data_source_name')
    
    parser.add_argument(
        '-did',
        '--drive_id',
        default=',')

    inputs=parser.parse_args()
    
    dataset_name=inputs.dataset_name
    data_source_name=inputs.data_source_name
    drive_id=inputs.drive_id

    load_dotenv()
    sql_d_o = c_sql_d_c(os.getenv('DB_HOST'), os.getenv('DB_PORT'), os.getenv('DB_NAME'), os.getenv('DB_USER'),
                                os.getenv('DB_PASSWORD'))

    # dropts='''
    # DROP TABLE Satelimgslocs;
    # '''
    # sql_d_c.process(dropts)
    
    # Define the SQL command to create a new table
    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS {dataset_name}(
    id SERIAL PRIMARY KEY,
    source VARCHAR(50),
    loc VARCHAR(100),
    label VARCHAR(50)
    );
    '''

    sql_d_o.process(create_table_query)
    
    
    
   
    
    zip_file_path = f"tmp/{data_source_name}.zip"
    
    # Google Drive file link
    url = f"https://drive.google.com/uc?id={drive_id}"


    try:
      # Download into memory or a temporary location
      gdown.download(url,zip_file_path, quiet=False)
    except:
      print('drive_id is wrong or empty')
      
    # Open the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # List all files in the ZIP
        file_names = z.namelist()
        
        # cmnds=[]
        # for file_name in file_names:
        #     if file_name.endswith(('.png', '.jpg', '.jpeg')):
                
        #         loc_com=f'''
        #         INSERT INTO Satelimgslocs(source,loc,label)
        #         VALUES
        #         ('{data_set_name}.zip','{file_names}','{file_name.split('/')[1]}')
        #         '''
        #         cmnds.append(loc_com)
                
        
        # with ThreadPoolExecutor() as executor:
            
        #     executor.map(sql_d_c.process,cmnds)
            
        loc_com=f'''
        INSERT INTO {dataset_name}(source,loc,label)
        VALUES
        
        '''
            
        for file_name in file_names:
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                loc_com+=f'''('{data_source_name}.zip','{file_name}','{file_name.split('/')[-2]}'),'''
                
        
        loc_com=loc_com[:-1]
                
        sql_d_o.process(loc_com)
        