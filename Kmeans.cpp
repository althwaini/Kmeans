//
//  Kmeans.cpp
//  imageLab
//
//  Created by kareem Althwaini on 2015-04-14.
//  Copyright (c) 2015 Kareem Althwaini. All rights reserved.
//

#include "Kmeans.h"
#include <iostream>

using namespace std;
using namespace cv;

Kmeans::Kmeans()
{
}

Kmeans::~Kmeans()
{
}

Kmeans::dst Kmeans::kmeans(vector<Image::src> src, int k, char method, float distFactor, int maxTries)
{
	int N = (int)src.size();
	int xs = src[0].mat.cols, ys = src[0].mat.rows;
	
	Mat *imgs = new Mat[N];
	
	
	for(int i = 0; i < N; i++)
		imgs[i] = src[i].mat;
	
	int level[N];
	for(int i = 0; i < N; i++)
		level[i] = src[i].level;
	
	src.clear();
	
	Kmeans::dst dst;
	dst.mat = Mat( ys, xs, CV_32S, Scalar(0));
	Mat dst_( ys, xs, CV_32S, Scalar(0));
	
	float centroid[k][N], centroid_[k][N], distant[k], d[N], change = 1.0;
	int changes = 0;
	
	int count[k]; float sum[k][N];
	float min = 1.0; int pos = 0;
	
	if (method == 'r' || method == 'R')
	{
		for(int j = 0; j < N; j++)
		{
			for(int i = 0; i < k ; i++)
				centroid[i][j] =  1.0 / (RAND_MAX) *  rand();
		}
	}
	else if (method == 'm' || method == 'M')
	{
		for(int j = 0; j < N; j++)
		{
			for(int i = 0; i < k ; i++)
				centroid[i][j] = float(i)/k + ((1.0/(float)k)/2.0);
		}
	}
	
	else if (method == 'd' || method == 'd')
	{
		
		for(int j = 0; j < N; j++)
		{
			vector<float> peaks = densityInit(imgs[j], k, level[j]);
			for(int i = 0; i < k ; i++)
				centroid[i][j] = peaks[i];
		}
	}
	
	else
	{
		printf ("You have to choose a valid method (R => Random, M => Median or D => Density)\n");
		exit (EXIT_FAILURE);
	}
	
	
	while(change > 0.0 && changes <= maxTries)
	{
		
		// initioal values with zeros
		for(int i = 0; i < k; i++)
		{
			count[i] = 0;
			for(int j = 0; j < N; j++)
			{
				sum[i][j] = 0.0;
			}
		}
		
		for(int j = 0; j < xs; j++)
		{
			for(int i = 0; i < ys; i++)
			{
				for(int l = 0; l < k; l++)
				{
					distant[l] = 0.0;
					
					for(int m = 0; m < N; m++)
					{
						if(level[m] == 179)
							imgs[m].at<float>(i,j) = abs(imgs[m].at<float>(i,j) - floor(imgs[m].at<float>(i,j) + 0.5));
						
						d[m] = abs(centroid[l][m]-imgs[m].at<float>(i,j));
						distant[l] += (d[m]*d[m]);
					}
					
					distant[l] = sqrt(distant[l]);
				}
				
				
				min = 1.0;
				pos = 0;
				
				for(int l = 0; l < k; l++)
				{
					if(distant[l] < min)
					{
						min = distant[l];
						pos = l;
					}
				}
				
				dst_.at<int>(i, j) = pos;
				count[pos]++;
				
				for(int l = 0; l < N; l++)
				{
					sum[pos][l] += imgs[l].at<float>(i, j);
				}
			}
		}
		
		
		change = 0.0;
		for(int i = 0; i < k; i++)
		{
			distant[i] = 0.0;
			for(int j = 0; j < N; j++)
			{
				centroid_[i][j] = sum[i][j] / (float)count[i];
				distant[i] += (abs(centroid[i][j]-centroid_[i][j]) * distFactor);
				centroid[i][j] += ((centroid_[i][j] - centroid[i][j]) * distFactor);
			}
			change += distant[i];
		}
		changes++;
	}
	
	vector<float> tmp;
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < k; j++)
			tmp.push_back(centroid[j][i]);
		//dst.centroids.push_back(tmp);
		tmp.clear();
	}
	
	dst.mat = dst_;
	dst.tries = changes;
	dst_.release();
	
	return dst;
}


vector<float> Kmeans::densityInit(Mat m, int n, int l)
{
	vector<float> clusters;
	
	Mat hist( 1, l+1, CV_32S, Scalar(0));
	for(int i = 0; i < m.rows; i++)
		for(int j = 0; j < m.cols; j++)
			if(m.at<float>(i,j) == m.at<float>(i,j))
				hist.at<int>(int(m.at<float>(i,j)*l))++;
	
	int sum = (m.cols*m.rows), dens = 0;
	dens = (float)sum / ((float)n+1.0);
	sum = 0.0;
	
	for(int i = 0; i < hist.cols; i++)
	{
		sum += hist.at<int>(i);
		if(sum >= (dens * clusters.size()+1))
		{
			clusters.push_back((float)i/ float(l));
		}
	}
	
	
	return clusters;
}
