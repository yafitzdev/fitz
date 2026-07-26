// api_client.ts
// TypeScript client for the InventoryHub REST API.
// Provides typed methods for product, warehouse, and shipment operations.

export interface Product {
  id: string;
  name: string;
  sku: string;
  category: string;
  priceUsd: number;
  stockCount: number;
}

export interface Warehouse {
  id: string;
  name: string;
  location: string;
  capacity: number;
  currentLoad: number;
}

export interface Shipment {
  id: string;
  warehouseId: string;
  productId: string;
  quantity: number;
  status: "pending" | "in_transit" | "delivered" | "cancelled";
  estimatedArrival: string;
}

export interface ApiError {
  code: number;
  message: string;
  details?: string;
}

/**
 * InventoryHubClient wraps the InventoryHub REST API.
 * All methods throw ApiClientError on non-2xx responses.
 */
export class InventoryHubClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.apiKey = apiKey;
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new ApiClientError(error.code, error.message, error.details);
    }

    return response.json() as Promise<T>;
  }

  /** GET /api/products — List all products, optionally filtered by category. */
  async listProducts(category?: string): Promise<Product[]> {
    const query = category ? `?category=${encodeURIComponent(category)}` : "";
    return this.request<Product[]>("GET", `/api/products${query}`);
  }

  /** GET /api/products/:id — Fetch a single product by ID. */
  async getProduct(productId: string): Promise<Product> {
    return this.request<Product>("GET", `/api/products/${productId}`);
  }

  /** POST /api/products — Create a new product entry. */
  async createProduct(product: Omit<Product, "id">): Promise<Product> {
    return this.request<Product>("POST", "/api/products", product);
  }

  /** GET /api/warehouses — List all warehouses. */
  async listWarehouses(): Promise<Warehouse[]> {
    return this.request<Warehouse[]>("GET", "/api/warehouses");
  }

  /** GET /api/warehouses/:id — Fetch warehouse details including current load. */
  async getWarehouse(warehouseId: string): Promise<Warehouse> {
    return this.request<Warehouse>("GET", `/api/warehouses/${warehouseId}`);
  }

  /** POST /api/shipments — Create a shipment from a warehouse. */
  async createShipment(
    warehouseId: string,
    productId: string,
    quantity: number
  ): Promise<Shipment> {
    return this.request<Shipment>("POST", "/api/shipments", {
      warehouseId,
      productId,
      quantity,
    });
  }

  /** GET /api/shipments/:id — Track a shipment's current status. */
  async getShipment(shipmentId: string): Promise<Shipment> {
    return this.request<Shipment>("GET", `/api/shipments/${shipmentId}`);
  }

  /** PATCH /api/shipments/:id/cancel — Cancel a pending shipment. */
  async cancelShipment(shipmentId: string): Promise<Shipment> {
    return this.request<Shipment>("PATCH", `/api/shipments/${shipmentId}/cancel`);
  }
}

export class ApiClientError extends Error {
  code: number;
  details?: string;

  constructor(code: number, message: string, details?: string) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.details = details;
  }
}
