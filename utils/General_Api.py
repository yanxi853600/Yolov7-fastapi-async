class Cal_IOU():

    def bb_intersection_over_union(boxA, boxB): 
        #  OFF_Line_WorkArea_boxA = [ OFF_Line_WorkArea_boxA['left'] * _WIDTH,  OFF_Line_WorkArea_boxA['top'] * _HEIGHT, _WIDTH *( OFF_Line_WorkArea_boxA['left']+   OFF_Line_WorkArea_boxA['width']), _HEIGHT *( OFF_Line_WorkArea_boxA['top']+  OFF_Line_WorkArea_boxA['height'] )]
        # boxB = [boxB['left'] * _WIDTH, boxB['top'] * _HEIGHT, _WIDTH *(boxB['left'] + boxB['width']), _HEIGHT *(boxB['top']+ boxB['height'] )]
        #  OFF_Line_WorkArea_boxA = [float(x) for x in  OFF_Line_WorkArea_boxA]
        # boxB = [float(x) for x in boxB]

        xA = max( boxA[0], boxB[0])
        yA = max( boxA[1], boxB[1])
        xB = min( boxA[2], boxB[2])
        yB = min( boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = ( boxA[2] -  boxA[0] + 1) * ( boxA[3] -  boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float( boxAArea + boxBArea - interArea)

        return iou 

    def bb_overlab(boxA, boxB):
        #  OFF_Line_WorkArea_boxA = [ OFF_Line_WorkArea_boxA['left'] * _WIDTH,  OFF_Line_WorkArea_boxA['top'] * _HEIGHT, _WIDTH *( OFF_Line_WorkArea_boxA['left']+   OFF_Line_WorkArea_boxA['width']), _HEIGHT *( OFF_Line_WorkArea_boxA['top']+  OFF_Line_WorkArea_boxA['height'] )]
        # boxB = [boxB['left'] * _WIDTH, boxB['top'] * _HEIGHT, _WIDTH *(boxB['left'] + boxB['width']), _HEIGHT *(boxB['top']+ boxB['height'] )]
        #  OFF_Line_WorkArea_boxA = [float(x) for x in  OFF_Line_WorkArea_boxA]
        # boxB = [float(x) for x in boxB]

        xA = max( boxA[0], boxB[0])
        yA = max( boxA[1], boxB[1])
        xB = min( boxA[2], boxB[2])
        yB = min( boxA[3], boxB[3])

        H = max(0, float(xB) - float(xA) + 1)
        W = max(0, float(yB) - float(yA) + 1)
        interArea = H * W
        
        # boxAArea = ( boxA[2] -  boxA[0] + 1) * ( boxA[3] -  boxA[1] + 1) 
        boxBArea = (float(boxB[2]) - float(boxB[0]) + 1) * (float(boxB[3]) - float(boxB[1]) + 1)
        
        over = interArea / float( boxBArea)
        
        return over

    def ObjectOverLap(Src_Object_list, CondLabelName_list , WorkArea):
        OverLap_List = []
        for Object in Src_Object_list:
            #print(Object['Label_Name'])
            if Object['Label_Name'] in CondLabelName_list:
                boxB = [ Object['X'], Object['y'], Object['u'], Object['v']]
                Overlap = Cal_IOU.bb_overlab( WorkArea, boxB)
                OverLap_List.append(Overlap)

        return OverLap_List
