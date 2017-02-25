#include "Conversion.hpp"

#include <gtest/gtest.h>

bool operator ==(const ram_msgs::Vector2 &a, const ram_msgs::Vector2 &b)
{
	return a.x == b.x && a.y == b.y;
}

bool operator ==(const ram_msgs::Vector3 &a, const ram_msgs::Vector3 &b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool operator ==(const ram_msgs::Vector4 &a, const ram_msgs::Vector4 &b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

bool operator ==(const ram_msgs::Matrix2 &a, const ram_msgs::Matrix2 &b)
{
	return a.matrix == b.matrix;
}

bool operator ==(const ram_msgs::Matrix3 &a, const ram_msgs::Matrix3 &b)
{
	return a.matrix == b.matrix;
}

bool operator ==(const ram_msgs::Matrix4 &a, const ram_msgs::Matrix4 &b)
{
	return a.matrix == b.matrix;
}

bool operator ==(const ram_msgs::Quaternion &a, const ram_msgs::Quaternion &b)
{
	return a.real == b.real && a.i == b.i && a.j == b.j && a.k == b.k;
}

bool operator ==(const ram_utils::Quaternion &a, const ram_utils::Quaternion &b)
{
	return a.w() == b.w() && a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
}

template<typename T, typename U>
void testConversion(const T& obj1, const U& obj2)
{
	U result1;
	T result2;

	ram_utils::convert(obj1, result1);
	ram_utils::convert(obj2, result2);

	EXPECT_TRUE(obj1 == result2);
	EXPECT_TRUE(obj2 == result1);
}

TEST(ConversionTests, vectorConversions)
{
	ram_msgs::Vector2 vec2;
	vec2.x = 1;
	vec2.y = 2;

	ram_msgs::Vector3 vec3;
	vec3.x = 1;
	vec3.y = 2;
	vec3.z = 3;

	ram_msgs::Vector4 vec4;
	vec4.x = 1;
	vec4.y = 2;
	vec4.z = 3;
	vec4.w = 4;

	testConversion(vec2, ram_utils::Vector2(1, 2));
	testConversion(vec2, ram_utils::RowVector2(1, 2));

	testConversion(vec3, ram_utils::Vector3(1, 2, 3));
	testConversion(vec3, ram_utils::RowVector3(1, 2, 3));

	testConversion(vec4, ram_utils::Vector4(1, 2, 3, 4));
	testConversion(vec4, ram_utils::RowVector4(1, 2, 3, 4));
}

TEST(ConversionTests, matrixConversions)
{
	ram_msgs::Matrix2 msg2;
	for(size_t i = 0; i < msg2.matrix.size(); ++i)
		msg2.matrix[i] = i + 1;

	ram_msgs::Matrix3 msg3;
	for(size_t i = 0; i < 9; ++i)
		msg3.matrix[i] = i + 1;

	ram_msgs::Matrix4 msg4;
	for(size_t i = 0; i < 16; ++i)
		msg4.matrix[i] = i + 1;

	ram_utils::Matrix2 mat2;
	mat2 << 1, 2,
			3, 4;

	ram_utils::Matrix3 mat3;
	mat3 << 1, 2, 3,
			4, 5, 6,
			7, 8, 9;

	ram_utils::Matrix4 mat4;
	mat4 << 1,  2,  3,  4,
			5,  6,  7,  8,
			9,  10, 11, 12,
			13, 14, 15, 16;

	testConversion(msg2, mat2);
	testConversion(msg3, mat3);
	testConversion(msg4, mat4);
}

TEST(ConversionTests, quaternionConversions)
{
	ram_msgs::Quaternion q;
	q.real = 1;
	q.i = 2;
	q.j = 3;
	q.k = 4;

	testConversion(q, ram_utils::Quaternion(1, 2, 3, 4));
}
