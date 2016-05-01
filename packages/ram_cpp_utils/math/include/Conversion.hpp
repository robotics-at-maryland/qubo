#pragma once

#include <ram_msgs/Matrix2.h>
#include <ram_msgs/Matrix3.h>
#include <ram_msgs/Matrix4.h>
#include <ram_msgs/Quaternion.h>
#include <ram_msgs/Vector2.h>
#include <ram_msgs/Vector3.h>
#include <ram_msgs/Vector4.h>

#include "Typedefs.hpp"

namespace ram_utils
{

void convert(const ram_msgs::Vector2& from, Vector2& to);
void convert(const ram_msgs::Vector3& from, Vector3& to);
void convert(const ram_msgs::Vector4& from, Vector4& to);

void convert(const ram_msgs::Vector2& from, RowVector2& to);
void convert(const ram_msgs::Vector3& from, RowVector3& to);
void convert(const ram_msgs::Vector4& from, RowVector4& to);

void convert(const ram_msgs::Matrix2& from, Matrix2& to);
void convert(const ram_msgs::Matrix3& from, Matrix3& to);
void convert(const ram_msgs::Matrix4& from, Matrix4& to);

void convert(const ram_msgs::Quaternion& from, Quaternion& to);

void convert(const Vector2& from, ram_msgs::Vector2& to);
void convert(const Vector3& from, ram_msgs::Vector3& to);
void convert(const Vector4& from, ram_msgs::Vector4& to);

void convert(const RowVector2& from, ram_msgs::Vector2& to);
void convert(const RowVector3& from, ram_msgs::Vector3& to);
void convert(const RowVector4& from, ram_msgs::Vector4& to);

void convert(const Matrix2& from, ram_msgs::Matrix2& to);
void convert(const Matrix3& from, ram_msgs::Matrix3& to);
void convert(const Matrix4& from, ram_msgs::Matrix4& to);

void convert(const Quaternion& from, ram_msgs::Quaternion& to);

}
