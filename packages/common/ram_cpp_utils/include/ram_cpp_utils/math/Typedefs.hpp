#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace ram_utils
{

typedef double Scalar;

typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
typedef Eigen::Matrix<Scalar, 4, 1> Vector4;

typedef Eigen::Matrix<Scalar, 1, 2> RowVector2;
typedef Eigen::Matrix<Scalar, 1, 3> RowVector3;
typedef Eigen::Matrix<Scalar, 1, 4> RowVector4;

typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;
typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;
typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

typedef Eigen::Quaternion<Scalar> Quaternion;

}
