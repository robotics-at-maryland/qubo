#include "Conversion.hpp"

namespace ram_utils
{

void convert(const ram_msgs::Vector2& from, Vector2& to)
{
	to.x() = from.x;
	to.y() = from.y;
}

void convert(const ram_msgs::Vector3& from, Vector3& to)
{
	to.x() = from.x;
	to.y() = from.y;
	to.z() = from.z;
}

void convert(const ram_msgs::Vector4& from, Vector4& to)
{
	to.x() = from.x;
	to.y() = from.y;
	to.z() = from.z;
	to.w() = from.w;
}

void convert(const ram_msgs::Vector2& from, RowVector2& to)
{
	to.x() = from.x;
	to.y() = from.y;
}

void convert(const ram_msgs::Vector3& from, RowVector3& to)
{
	to.x() = from.x;
	to.y() = from.y;
	to.z() = from.z;
}

void convert(const ram_msgs::Vector4& from, RowVector4& to)
{
	to.x() = from.x;
	to.y() = from.y;
	to.z() = from.z;
	to.w() = from.w;
}

void convert(const ram_msgs::Matrix2& from, Matrix2& to)
{
	for(int i = 0; i < 4; ++i)
		to(i/2, i%2) = from.matrix[i];
}

void convert(const ram_msgs::Matrix3& from, Matrix3& to)
{
	for(int i = 0; i < 9; ++i)
		to(i/3, i%3) = from.matrix[i];
}

void convert(const ram_msgs::Matrix4& from, Matrix4& to)
{
	for(int i = 0; i < 16; ++i)
		to(i/4, i%4) = from.matrix[i];
}

void convert(const ram_msgs::Quaternion& from, Quaternion& to)
{
	to.w() = from.real;
	to.x() = from.i;
	to.y() = from.j;
	to.z() = from.k;
}

void convert(const Vector2& from, ram_msgs::Vector2& to)
{
	to.x = from.x();
	to.y = from.y();
}

void convert(const Vector3& from, ram_msgs::Vector3& to)
{
	to.x = from.x();
	to.y = from.y();
	to.z = from.z();
}

void convert(const Vector4& from, ram_msgs::Vector4& to)
{
	to.x = from.x();
	to.y = from.y();
	to.z = from.z();
	to.w = from.w();
}

void convert(const RowVector2& from, ram_msgs::Vector2& to)
{
	to.x = from.x();
	to.y = from.y();
}

void convert(const RowVector3& from, ram_msgs::Vector3& to)
{
	to.x = from.x();
	to.y = from.y();
	to.z = from.z();
}

void convert(const RowVector4& from, ram_msgs::Vector4& to)
{
	to.x = from.x();
	to.y = from.y();
	to.z = from.z();
	to.w = from.w();
}

void convert(const Matrix2& from, ram_msgs::Matrix2& to)
{
	for(int i = 0; i < 4; ++i)
		to.matrix[i] = from(i/2, i%2);
}

void convert(const Matrix3& from, ram_msgs::Matrix3& to)
{
	for(int i = 0; i < 9; ++i)
		to.matrix[i] = from(i/3, i%3);
}

void convert(const Matrix4& from, ram_msgs::Matrix4& to)
{
	for(int i = 0; i < 16; ++i)
		to.matrix[i] = from(i/4, i%4);
}

void convert(const Quaternion& from, ram_msgs::Quaternion& to)
{
	to.real = from.w();
	to.i = from.x();
	to.j = from.y();
	to.k = from.z();
}

}
