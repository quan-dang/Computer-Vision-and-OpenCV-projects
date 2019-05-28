import sys
import numpy as np
import cv2

# read points from text file
def readPoints(path):
    # create an array of points
    points = []

    # read points 
    with open(path) as f:
        for line in f:
            x, y = line.split()
            points.append((int(x), int(y)))
        

    return points


# apply affine transform calculated using srcTri and dstTri and return an image of size
def applyAffineTransform(src, srcTri, dstTri, size):
    # given a pair of triangles, find the affine transform
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # apply the affine transform above to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# check if a point inside a rectangle
def rectContains(rect, point):
    if (point[0] < rect[0]):
        return False
    elif (point[1] < rect[1]):
        return False
    elif (point[0] > rect[0] + rect[2]):
        return False
    elif (point[1] > rect[1] + rect[3]):
        return False
    
    return True


# create delaunay triangles
def calculateDelaunayTriangles(rect, points):
    # creat subdiv
    subdiv = cv2.Subdiv2D(rect)

    # insert points into subdiv
    for point in points:
        subdiv.insert(point)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            inds = []

            # get face-points from 68 face detector by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0) and abs(pt[j][1] - points[k][1] < 1.0):
                        inds.append(k)

            # three points form a triangle
            # triangle array corresponds to the file tri.txt in FaceMorph
            if len(inds) == 3:
                delaunayTri.append((inds[0], inds[1], inds[2]))

        pt = []

        
    return delaunayTri


# warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # offset points by left-top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))


    # get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]: r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    # copy triangular region of the rectangular patch to the output image
    img2[r2[1]: r2[1] + r2[3], r2[0]: r2[0] + r2[2]] = img2[r2[1]: r2[1] + r2[3], r2[0]: r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]: r2[1] + r2[3], r2[0]: r2[0] + r2[2]] = img2[r2[1]: r2[1] + r2[3], r2[0]: r2[0] + r2[2]] + img2Rect


if __name__ == '__main__':
    # make sure using OpenCv version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        print("Error: OpenCV 3.0 or higher required!")
        sys.exit(1)

    # read images
    filename1 = 'hillary_clinton.jpg'
    filename2 = 'donald_trump.jpg'

    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    img1Warped = np.copy(img2)

    # read array of correspoding points
    points1 = readPoints(filename1 + '.txt') 
    points2 = readPoints(filename2 + '.txt')

    # find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])


    # find delaunay triangulation for convex hull points
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = calculateDelaunayTriangles(rect, hull2)

    if len(dt) == 0:
        quit()

    # apply affine transformation to delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warpTriangle(img1, img1Warped, t1, t2)

    
    # calculate mask
    hull8U = []

    for i in range(0, len(hull2)):
        hull8U.append((hull1[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0] + int(r[2]/2), r[1]+int(r[3]/2)))

    # clone seamlessly
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

    cv2.imshow("Face Swapped", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

